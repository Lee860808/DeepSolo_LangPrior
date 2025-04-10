"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from adet.utils.curve_utils import BezierSampler
from .linguistic_prior_utils import generate_soft_target_sequence

class CtrlPointHungarianMatcher(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            text_weight: float = 1,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            centroids: torch.Tensor = None,
            vocab: list = None,
            text_cost_type: str = 'kldiv_avg', # Option for cost calculation
            soft_target_threshold: float = 0.85,
    ):
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.text_weight = text_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"
        if text_weight > 0:
             assert centroids is not None and vocab is not None, \
                 "Centroids and vocab must be provided if text_weight > 0"
        self.centroids = centroids # Shape: (voc_size, embed_dim)
        self.vocab = vocab
        self.voc_size = len(vocab) if vocab else 0
        self.soft_target_threshold = soft_target_threshold
        self.text_cost_type = text_cost_type
        # Add epsilon for KL divergence stability
        self.kldiv_epsilon = 1e-6

    def calculate_linguistic_cost(self, out_texts_prob_batch, target_texts_indices_batch,target_lengths_batch, centroids, vocab, threshold):
        num_queries, n_pts, _ = out_texts_prob_batch.shape
        num_gt = target_texts_indices_batch.shape[0]
        device = out_texts_prob_batch.device
        cost_matrix = torch.zeros(num_queries, num_gt, device=device)

        if num_gt == 0:
            return cost_matrix

        # 1. Generate all soft target sequences for the batch item
        flat_target_indices = torch.cat([t[:l] for t, l in zip(target_texts_indices_batch, target_lengths_batch)])
        soft_target_sequences = generate_soft_target_sequence(
            flat_target_indices, target_lengths_batch, centroids, vocab, threshold
        ) # List of [L_i, voc_size] tensors

        # 2. Prepare predicted distributions (Average over points)
        # Use probs for voc_size chars (exclude blank/unknown if present)
        pred_prob_avg = out_texts_prob_batch[..., :self.voc_size].mean(dim=1) # (nq, voc_size)

        # 3. Calculate cost for each query-target pair
        for i in range(num_queries):
            pred_dist_i = pred_prob_avg[i] # (voc_size,)
            for j in range(num_gt):
                if target_lengths_batch[j] == 0:
                     # Assign high cost if GT is empty? Or skip? Let's use high cost.
                     cost_matrix[i, j] = 100.0 # Arbitrary high cost
                     continue

                target_dist_seq_j = soft_target_sequences[j] # (L_j, voc_size)
                # Average target distributions
                target_dist_avg_j = target_dist_seq_j.mean(dim=0) # (voc_size,)

                # Calculate cost (e.g., KL Divergence D_KL(Target || Pred))
                # Ensure predicted probs are non-zero for stability
                pred_dist_i_stable = pred_dist_i + self.kldiv_epsilon
                pred_dist_i_stable /= pred_dist_i_stable.sum()

                # Ensure target probs sum to 1 (should be close after thresholding)
                target_dist_avg_j_stable = target_dist_avg_j + self.kldiv_epsilon
                target_dist_avg_j_stable /= target_dist_avg_j_stable.sum()

                # KL divergence
                kl_cost = F.kl_div(pred_dist_i_stable.log(), target_dist_avg_j_stable, reduction='sum')
                # print(f"KL Cost q{i}-t{j}: {kl_cost.item()}") # Debugging

                # Optional: Use L1 distance instead (maybe more stable?)
                # l1_cost = torch.cdist(pred_dist_i.unsqueeze(0), target_dist_avg_j.unsqueeze(0), p=1).squeeze()
                # cost_matrix[i, j] = l1_cost

                cost_matrix[i, j] = kl_cost

        return cost_matrix
        

    def forward(self, outputs, targets):
        with torch.no_grad():
            sizes = [len(v["ctrl_points"]) for v in targets]
            bs, num_queries = outputs["pred_logits"].shape[:2]

            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()

            out_prob_avg = out_prob.mean(1) # (bs * nq, num_classes)
            tgt_ids = torch.cat([v["labels"] for v in targets]) # (total_num_gt,)

            out_texts = F.log_softmax(outputs['pred_text_logits'], dim=-1)  # (bs, n_q, n_pts, voc+1)
            n_pts, voc = out_texts.shape[2], out_texts.shape[-1] - 1
            target_texts = torch.cat([v["texts"] for v in targets])
            target_lengths = (target_texts != voc).long().sum(dim=-1)
            target_texts = torch.split(target_texts, sizes, dim=0)
            target_lengths = torch.split(target_lengths, sizes)
            texts_cost_list = []
            for out_texts_batch, targe_texts_batch, target_len_batch in zip(out_texts, target_texts, target_lengths):
                out_texts_batch_temp = out_texts_batch.repeat(targe_texts_batch.shape[0], 1, 1).permute(1, 0, 2)
                input_len = torch.full((out_texts_batch_temp.size(1),), out_texts_batch_temp.size(0), dtype=torch.long)
                targe_texts_batch_temp = torch.cat([
                    t[:target_len_batch[t_idx]].repeat(num_queries) for t_idx, t in enumerate(targe_texts_batch)
                ])
                target_len_batch_temp = target_len_batch.reshape((-1, 1)).repeat(1, num_queries).reshape(-1)
                text_cost = F.ctc_loss(
                    out_texts_batch_temp,
                    targe_texts_batch_temp,
                    input_len,
                    target_len_batch_temp,
                    blank=voc,
                    zero_infinity=True,
                    reduction='none'
                )
                text_cost.div_(target_len_batch_temp)
                text_cost_cpu = text_cost.reshape((-1, num_queries)).transpose(1, 0).cpu()
                texts_cost_list.append(text_cost_cpu)

            neg_cost_class = (1 - self.alpha) * (out_prob_avg ** self.gamma) * \
                (-(1 - out_prob_avg + 1e-8).log())
            pos_cost_class = self.alpha * \
                ((1 - out_prob_avg) ** self.gamma) * (-(out_prob_avg + 1e-8).log())
            # Match against the "text" class (index 0, assuming binary)
            cost_class = pos_cost_class[:, 0] - neg_cost_class[:, 0] # (bs * nq,) - use index 0 for text class    

            # ctrl points of the text center line: (bz, n_q, n_pts, 2) --> (bz * n_q, n_pts * 2)
            out_pts = outputs["pred_ctrl_points"].flatten(0, 1).flatten(-2)
            tgt_pts = torch.cat([v["ctrl_points"] for v in targets]).flatten(-2)

            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * \
                ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = (pos_cost_class[..., 0] - neg_cost_class[..., 0]).mean(-1, keepdims=True)
            cost_kpts = torch.cdist(out_pts, tgt_pts, p=1)  # (bz * n_q, num_gt)
            
            cost_text = torch.zeros_like(cost_kpts) # Initialize
            if self.text_weight > 0 and self.centroids is not None:
                # Use predicted LOGITS for stability before softmax/log_softmax
                out_texts_logits = outputs['pred_text_logits'] # (bs, nq, n_pts, voc + 1)
                out_texts_prob = F.softmax(out_texts_logits, dim=-1) # (bs, nq, n_pts, voc+1)

                # Target texts need careful handling for batching
                target_texts_list = [v["texts"] for v in targets] # List of [num_gt_i, max_len]
                target_lengths_list = [(t != self.voc_size).long().sum(dim=-1) for t in target_texts_list] # List of [num_gt_i]

                start_idx = 0
                for i in range(bs): # Process each batch item
                    num_gt_i = sizes[i]
                    if num_gt_i == 0:
                        continue

                    end_idx = start_idx + num_gt_i
                    cost_text_batch = self.calculate_linguistic_cost(
                        out_texts_prob[i],            # (nq, n_pts, voc+1)
                        target_texts_list[i],       # (num_gt_i, max_len)
                        target_lengths_list[i],     # (num_gt_i,)
                        self.centroids,
                        self.vocab,
                        self.soft_target_threshold
                    ) # Should return shape (nq, num_gt_i)

                    # Fill the corresponding part of the global cost matrix
                    cost_text[i * num_queries : (i + 1) * num_queries, start_idx : end_idx] = cost_text_batch
                    start_idx = end_idx
            else:
                 print("Skipping linguistic text cost calculation.")


            # --- Final Cost ---
            # Reshape class cost to match kpts and text cost
            cost_class = cost_class.unsqueeze(1).repeat(1, cost_kpts.shape[1])

            C = (self.class_weight * cost_class +
                 self.coord_weight * cost_kpts +
                 self.text_weight * cost_text)

            C = C.view(bs, num_queries, -1).cpu()

            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

            C = self.class_weight * cost_class + self.coord_weight * cost_kpts
            C = C.view(bs, num_queries, -1).cpu()

            indices = [linear_sum_assignment(
                c[i] + self.text_weight * texts_cost_list[i]
            ) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class BezierHungarianMatcher(nn.Module):
    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            num_sample_points: int = 100,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        """Creates the matcher
        Params:
            class_weight: This is the relative weight of the classification error in the matching cost
            coord_weight: not the control points of bezier curve but the sampled points on curve,
            refer to "https://github.com/voldemortX/pytorch-auto-drive"
        """
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.num_sample_points = num_sample_points
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points)

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_beziers = outputs["pred_beziers"].flatten(0, 1).view(-1, 4, 2)  # (batch_size * num_queries, 4, 2)

            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_beziers = torch.cat([v["beziers"] for v in targets])  # (g, 4, 2)

            # Compute the classification cost.
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                             (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * \
                             ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost betweeen sampled points on Bezier curve
            cost_coord = torch.cdist(
                (self.bezier_sampler.get_sample_points(out_beziers)).flatten(start_dim=-2),
                (self.bezier_sampler.get_sample_points(tgt_beziers)).flatten(start_dim=-2),
                p=1
            )

            C = self.class_weight * cost_class + self.coord_weight * cost_coord
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["beziers"]) for v in targets]
            indices = [
                linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
            ]

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cfg,centroids=None, vocab=None):
    cfg = cfg.MODEL.TRANSFORMER.LOSS
    return BezierHungarianMatcher(class_weight=cfg.BEZIER_CLASS_WEIGHT,
                                  coord_weight=cfg.BEZIER_COORD_WEIGHT,
                                  num_sample_points=cfg.BEZIER_SAMPLE_POINTS,
                                  focal_alpha=cfg.FOCAL_ALPHA,
                                  focal_gamma=cfg.FOCAL_GAMMA), \
           CtrlPointHungarianMatcher(class_weight=cfg.POINT_CLASS_WEIGHT,
                                     coord_weight=cfg.POINT_COORD_WEIGHT,
                                     text_weight=cfg.POINT_TEXT_WEIGHT,
                                     focal_alpha=cfg.FOCAL_ALPHA,
                                     focal_gamma=cfg.FOCAL_GAMMA,
                                     centroids=centroids,
                                     vocab=vocab,)