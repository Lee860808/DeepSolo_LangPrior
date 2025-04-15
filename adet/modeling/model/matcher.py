# DeepSolo_LangPrior/DeepSolo/adet/modeling/model/matcher.py
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from adet.utils.curve_utils import BezierSampler

# Import the updated utils
from .linguistic_prior_utils import generate_soft_target_sequence
import warnings # For warnings

class BezierHungarianMatcher(nn.Module):
    """Matches proposals with ground truth for Bezier curves (used by encoder)."""
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            num_sample_points: int = 100,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.num_sample_points = num_sample_points
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points)

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # (bs * num_queries, num_classes)
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        # (bs * num_queries, 8) -> (bs * num_queries, 4, 2)
        out_beziers = outputs["pred_beziers"].flatten(0, 1).view(-1, 4, 2)

        # Also concat the target labels and boxes
        # Assuming target labels are 0 for text, num_classes for background
        # tgt_ids: (total_num_gt,)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        # tgt_beziers: (total_num_gt, 4, 2)
        tgt_beziers = torch.cat([v["beziers"] for v in targets])

        # Compute the classification cost.
        neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
        # Cost is computed only for the foreground class (index 0)
        cost_class = pos_cost_class[:, 0] - neg_cost_class[:, 0] # Shape: (bs * num_queries,)

        # Compute the L1 cost betweeen sampled points on Bezier curve
        # Sample points: (N, num_sample_points, 2) -> flatten -> (N, num_sample_points * 2)
        out_samples = self.bezier_sampler.get_sample_points(out_beziers).flatten(start_dim=-2)
        tgt_samples = self.bezier_sampler.get_sample_points(tgt_beziers).flatten(start_dim=-2)
        cost_coord = torch.cdist(out_samples, tgt_samples, p=1) # (bs * num_queries, total_num_gt)

        # Ensure cost_class is broadcastable
        cost_class = cost_class.unsqueeze(1).repeat(1, cost_coord.shape[1]) # (bs * num_queries, total_num_gt)

        # Final cost matrix
        C = self.class_weight * cost_class + self.coord_weight * cost_coord
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["beziers"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class CtrlPointHungarianMatcher(nn.Module):
    """Matches proposals with ground truth for Control Points (used by decoder)."""
    def __init__(
            self,
            cfg, # Pass cfg
            class_weight: float = 1,
            coord_weight: float = 1,
            text_weight: float = 1, # Weight for the new linguistic cost
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            centroids: torch.Tensor = None,
            vocab: list = None,
            # text_cost_type is illustrative, only 'kldiv_avg' implemented below
            text_cost_type: str = 'kldiv_avg',
    ):
        super().__init__()
        self.cfg = cfg # Store cfg
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.text_weight = text_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0 or text_weight != 0, "all costs cant be 0"

        if text_weight > 0:
            assert centroids is not None and vocab is not None, \
                "Centroids and vocab must be provided if text_weight > 0"
        self.centroids = centroids
        self.vocab = vocab
        self.char_voc_size = len(vocab) if vocab else 0
        self.text_cost_type = text_cost_type
        self.kldiv_epsilon = 1e-6

    @torch.no_grad()
    def calculate_linguistic_cost(self, cfg, out_texts_prob_batch, target_texts_indices_batch,
                                 target_lengths_batch, centroids, vocab):
        """
        Calculates linguistic cost between predicted distributions and soft targets (BATCH version).
        Args:
            cfg (CfgNode): Config object
            out_texts_prob_batch (Tensor): Predicted probs (nq, n_pts, voc_size + 1). Needs vocab_size.
            target_texts_indices_batch (Tensor): GT indices (num_gt, max_len).
            target_lengths_batch (Tensor): GT lengths (num_gt,).
            centroids (Tensor): Character centroids (char_voc_size, embed_dim).
            vocab (list): Character list (size=char_voc_size).
        Returns:
            Tensor: Cost matrix (nq, num_gt).
        """
        num_queries, n_pts, _ = out_texts_prob_batch.shape
        num_gt = target_texts_indices_batch.shape[0]
        device = out_texts_prob_batch.device
        cost_matrix = torch.full((num_queries, num_gt), 100.0, device=device) # Initialize high

        if num_gt == 0 or self.char_voc_size == 0:
            return cost_matrix # Return high cost if no GT or no vocab

        # 1. Generate all soft target sequences for the batch item
        # Ensure targets indices are valid (less than char_voc_size) before generating targets
        flat_target_indices = torch.cat([t[:l] for t, l in zip(target_texts_indices_batch, target_lengths_batch)])

        # Pass cfg to generate_soft_target_sequence
        soft_target_sequences = generate_soft_target_sequence(
            cfg, flat_target_indices, target_lengths_batch, centroids, vocab
        ) # List of [L_i, char_voc_size] tensors, L_i can be 0

        # 2. Prepare predicted distributions (Average over points)
        # Slice to get probs for actual characters, excluding the last (blank/unknown) class
        pred_prob_avg = out_texts_prob_batch[..., :self.char_voc_size].mean(dim=1) # (nq, char_voc_size)

        # 3. Calculate cost for each query-target pair
        for i in range(num_queries):
            pred_dist_i = pred_prob_avg[i] # (char_voc_size,)
            for j in range(num_gt):
                if target_lengths_batch[j] == 0 or soft_target_sequences[j].numel() == 0:
                     # Keep high cost if GT is empty
                     continue

                target_dist_seq_j = soft_target_sequences[j] # (L_j, char_voc_size)
                # Average target distributions
                target_dist_avg_j = target_dist_seq_j.mean(dim=0) # (char_voc_size,)

                # --- Cost Calculation ---
                if self.text_cost_type == 'kldiv_avg':
                    # Ensure stability for KL divergence
                    pred_dist_i_stable = pred_dist_i.clamp(min=self.kldiv_epsilon)
                    # Target should already be normalized, but enforce for safety
                    target_dist_avg_j_stable = target_dist_avg_j.clamp(min=self.kldiv_epsilon)
                    target_dist_avg_j_stable /= target_dist_avg_j_stable.sum()

                    # KL divergence D_KL(Target || Pred) - expects log-probabilities for input
                    kl_cost = F.kl_div(pred_dist_i_stable.log(), target_dist_avg_j_stable, reduction='sum', log_target=False)
                    cost_matrix[i, j] = kl_cost.clamp(min=0) # Ensure non-negative cost
                # Add elif for other cost types ('l1', 'cosine', etc.) if implemented
                else:
                    raise NotImplementedError(f"Text cost type '{self.text_cost_type}' not implemented.")

        return cost_matrix

    @torch.no_grad()
    def forward(self, outputs, targets):
        sizes = [len(v["ctrl_points"]) for v in targets]
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # --- Class Cost ---
        # (bs * nq, n_pts, num_classes) -> (bs * nq, num_classes) by averaging over points
        out_prob_avg = outputs["pred_logits"].flatten(0, 1).sigmoid().mean(1)
        tgt_ids = torch.cat([v["labels"] for v in targets]) # (total_num_gt,)

        neg_cost_class = (1 - self.alpha) * (out_prob_avg ** self.gamma) * (-(1 - out_prob_avg + 1e-8).log())
        pos_cost_class = self.alpha * ((1 - out_prob_avg) ** self.gamma) * (-(out_prob_avg + 1e-8).log())
        #Foreground class is index 0
        cost_class = pos_cost_class[:, 0] - neg_cost_class[:, 0] # (bs * nq,)

        # --- Coordinate Cost ---
        # (bs * nq, n_pts * 2)
        out_pts = outputs["pred_ctrl_points"].flatten(0, 1).flatten(-2)
        # (total_num_gt, n_pts * 2)
        tgt_pts = torch.cat([v["ctrl_points"] for v in targets]).flatten(-2)
        cost_kpts = torch.cdist(out_pts, tgt_pts, p=1) # (bs * nq, total_num_gt)

        # --- Linguistic Text Cost ---
        cost_text = torch.zeros_like(cost_kpts)
        if self.text_weight > 0 and self.centroids is not None and self.vocab:
            out_texts_logits = outputs['pred_text_logits'] # (bs, nq, n_pts, voc + 1)
            out_texts_prob = F.softmax(out_texts_logits, dim=-1) # Use probabilities for cost

            target_texts_list = [v["texts"] for v in targets]
            # Get model's voc_size (inc blank) from logits, GT uses this convention too
            model_voc_size = out_texts_logits.shape[-1] - 1
            target_lengths_list = [(t != model_voc_size).long().sum(dim=-1) for t in target_texts_list]

            start_idx_gt = 0
            for i in range(bs): # Process each batch item
                num_gt_i = sizes[i]
                if num_gt_i == 0: continue
                end_idx_gt = start_idx_gt + num_gt_i

                cost_text_batch = self.calculate_linguistic_cost(
                    self.cfg, # Pass cfg
                    out_texts_prob[i],          # (nq, n_pts, voc+1)
                    target_texts_list[i],       # (num_gt_i, max_len)
                    target_lengths_list[i],     # (num_gt_i,)
                    self.centroids,             # (char_voc_size, embed_dim)
                    self.vocab                  # list, len=char_voc_size
                ) # Returns (nq, num_gt_i)

                query_start_idx = i * num_queries
                query_end_idx = (i + 1) * num_queries
                cost_text[query_start_idx : query_end_idx, start_idx_gt : end_idx_gt] = cost_text_batch
                start_idx_gt = end_idx_gt
        # else: # Optional: print only once
        #     if not hasattr(self, '_warned_skip_text_cost'):
        #         print("Skipping linguistic text cost calculation in matcher.")
        #         self._warned_skip_text_cost = True

        # --- Final Cost ---
        cost_class = cost_class.unsqueeze(1).expand_as(cost_kpts) # Expand class cost
        C = (self.class_weight * cost_class +
             self.coord_weight * cost_kpts +
             self.text_weight * cost_text) # Add linguistic cost

        C = C.view(bs, num_queries, -1).cpu()

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# Updated build_matcher function
def build_matcher(cfg, centroids=None, vocab=None):
    loss_cfg = cfg.MODEL.TRANSFORMER.LOSS
    bezier_matcher = BezierHungarianMatcher(
        class_weight=loss_cfg.BEZIER_CLASS_WEIGHT,
        coord_weight=loss_cfg.BEZIER_COORD_WEIGHT,
        num_sample_points=loss_cfg.BEZIER_SAMPLE_POINTS,
        focal_alpha=loss_cfg.FOCAL_ALPHA,
        focal_gamma=loss_cfg.FOCAL_GAMMA
    )
    point_matcher = CtrlPointHungarianMatcher(
        cfg=cfg, # Pass the full cfg here
        class_weight=loss_cfg.POINT_CLASS_WEIGHT,
        coord_weight=loss_cfg.POINT_COORD_WEIGHT,
        text_weight=loss_cfg.POINT_TEXT_WEIGHT, # This weight enables/disables linguistic cost
        focal_alpha=loss_cfg.FOCAL_ALPHA,
        focal_gamma=loss_cfg.FOCAL_GAMMA,
        centroids=centroids,
        vocab=vocab,
        # text_cost_type='kldiv_avg', # Could be made configurable
    )
    return bezier_matcher, point_matcher