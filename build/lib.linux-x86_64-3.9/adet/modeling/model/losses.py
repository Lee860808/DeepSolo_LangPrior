import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import json
import os
from tqdm import tqdm
from transformers import CanineTokenizer, CanineModel
from detectron2.data import MetadataCatalog
import copy
from adet.utils.misc import accuracy, is_dist_avail_and_initialized
from detectron2.utils.comm import get_world_size
from adet.utils.curve_utils import BezierSampler
from .matcher import build_matcher # Keep original for bezier
#from .matcher_langprior import build_matcher as build_matcher_langprior # Your modified matcher file/func name
#from .linguistic_prior_utils import load_char_embeddings, generate_centroids, get_corpus
from .linguistic_prior_utils import get_char_embeddings_canine, generate_centroids, get_corpus # Import specific functions
import pickle # For loading custom dict

_canine_model_cache = {}
_canine_tokenizer_cache = {}

# Pass cfg to get model name
def get_canine_model(cfg, device='cpu'):
    """Loads CANINE model and tokenizer, using a cache."""
    global _canine_model_cache, _canine_tokenizer_cache
    model_name = cfg.MODEL.LINGUISTIC_PRIOR.EMBED_MODEL_NAME # Read from cfg
    if model_name not in _canine_model_cache:
        print(f"Loading CANINE model: {model_name}...")
        model = CanineModel.from_pretrained(model_name).to(device)
        model.eval() # Set to evaluation mode
        _canine_model_cache[model_name] = model
        print(f"Loading CANINE tokenizer: {model_name}...")
        tokenizer = CanineTokenizer.from_pretrained(model_name)
        _canine_tokenizer_cache[model_name] = tokenizer
    return _canine_model_cache[model_name], _canine_tokenizer_cache[model_name]

def get_char_embeddings_canine(cfg, characters, device='cpu'):
    """
    Computes embeddings for a list of characters using CANINE.
    Args:
        cfg (CfgNode): Configuration object.
        characters (list): List of unique characters.
        device: Torch device.
    Returns:
        dict: Mapping character -> embedding tensor.
    """
    model_name = cfg.MODEL.LINGUISTIC_PRIOR.EMBED_MODEL_NAME
    model, tokenizer = get_canine_model(cfg, device) # Pass cfg
    embeddings_dict = {}
    print(f"Generating character embeddings using {model_name}...")

    for char in tqdm(characters):
        inputs = tokenizer(char, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        token_ids = inputs['input_ids'][0].tolist()
        first_char_token_idx = -1
        for idx, token_id in enumerate(token_ids):
             if token_id not in tokenizer.all_special_ids:
                 first_char_token_idx = idx
                 break

        if first_char_token_idx != -1:
            char_embedding = outputs.last_hidden_state[0, first_char_token_idx, :].detach().cpu()
            embeddings_dict[char] = char_embedding
        else:
            warnings.warn(f"Could not find valid character token for '{char}' in CANINE output. Skipping.")
            embeddings_dict[char] = None

    print("Character embedding generation complete.")
    return embeddings_dict

# Pass cfg to get dictionary path
def get_corpus(cfg, dataset_names):
    """
    Loads words from training datasets annotations and an optional generic dictionary.
    Args:
        cfg (CfgNode): Configuration object.
        dataset_names (tuple): Names of training datasets registered in Detectron2.
    Returns:
        list: A list of unique words making up the corpus (converted to uppercase).
    """
    print("Building corpus...")
    corpus_words = set()
    generic_dict_path = cfg.MODEL.LINGUISTIC_PRIOR.GENERIC_DICT_PATH # Read from cfg

    # 1. Load from dataset annotations (Logic remains the same)
    for dataset_name in dataset_names:
        try:
            metadata = MetadataCatalog.get(dataset_name)
            json_file = metadata.json_file
            print(f"  Loading annotations from: {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f: # Added encoding
                data = json.load(f)
                if 'annotations' in data:
                    for ann in data['annotations']:
                        text = None
                        if 'rec' in ann and isinstance(ann['rec'], str):
                            text = ann['rec']
                        elif 'text' in ann and isinstance(ann['text'], str): # Common key in some formats
                             text = ann['text']
                        elif 'utf8_string' in ann and isinstance(ann['utf8_string'], str): # Another common key
                             text = ann['utf8_string']
                        # Add other potential keys if datasets use different names for transcription

                        if text:
                             word = text.strip()
                             # Basic filtering (optional): remove very short/long or non-alphanumeric heavy words
                             if word and len(word) > 1 and len(word) < 30: # Example filter
                                 corpus_words.add(word.upper())

                else:
                     print(f"    Warning: No 'annotations' key found in {json_file}")
        except FileNotFoundError:
             print(f"  Warning: Annotation file not found: {json_file}. Skipping.")
        except Exception as e:
            print(f"  Warning: Could not load or parse annotations for {dataset_name}. Error: {e}")


    # 2. Load from generic dictionary (Logic remains the same)
    if generic_dict_path and os.path.exists(generic_dict_path):
        print(f"  Loading generic dictionary from: {generic_dict_path}")
        try:
            with open(generic_dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        corpus_words.add(word.upper())
        except Exception as e:
            print(f"  Warning: Could not load generic dictionary. Error: {e}")
    else:
        print(f"  Warning: Generic dictionary path not found or not specified: {generic_dict_path}")

    corpus_list = sorted(list(corpus_words))
    print(f"Corpus built with {len(corpus_list)} unique words.")
    return corpus_list

def sigmoid_focal_loss(inputs, targets, num_inst, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss.ndim == 4:
        return loss.mean((1, 2)).sum() / num_inst
    elif loss.ndim == 3:
        return loss.mean(1).sum() / num_inst
    else:
        raise NotImplementedError(f"Unsupported dim {loss.ndim}")


class SetCriterion(nn.Module):
    """
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
            self,
            cfg, # Pass the whole config object
            num_classes,
            # enc_matcher, # Will build inside
            # dec_matcher, # Will build inside
            weight_dict,
            enc_losses,
            num_sample_points,
            dec_losses,
            voc_size, # Keep for CTC loss
            num_ctrl_points,
            focal_alpha=0.25,
            focal_gamma=2.0
    ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.cfg = cfg # Store config
        self.num_classes = num_classes
        #self.enc_matcher = enc_matcher
        #self.dec_matcher = dec_matcher
        self.weight_dict = weight_dict
        self.enc_losses = enc_losses
        self.num_sample_points = num_sample_points
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points)
        self.dec_losses = dec_losses
        self.voc_size = voc_size
        self.char_voc_size = voc_size - 1 # Actual number of character classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_ctrl_points = num_ctrl_points
        self.text_match_weight = cfg.MODEL.TRANSFORMER.LOSS.POINT_TEXT_WEIGHT
        self.centroids = None
        self.vocab = None
        if self.text_match_weight > 0:
            print("Initializing Linguistic Priors for Matcher...")
            # Define vocab (ensure it matches your model's output layer size - 1)
            device = torch.device(cfg.MODEL.DEVICE)
            if self.voc_size == 37:
                self.vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            elif self.voc_size == 96:
                 self.vocab = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
            elif cfg.MODEL.TRANSFORMER.CUSTOM_DICT:
                 # TODO: Load vocab from CUSTOM_DICT file correctly
                 import pickle
                 try:
                     with open(cfg.MODEL.TRANSFORMER.CUSTOM_DICT, 'rb') as fp:
                         custom_vocab_list = pickle.load(fp)
                     # Convert integer char codes back to characters if needed
                     self.vocab = [chr(c) for c in custom_vocab_list] # Example conversion
                     print(f"Loaded custom vocab of size {len(self.vocab)} from {cfg.MODEL.TRANSFORMER.CUSTOM_DICT}")
                 except Exception as e:
                     print(f"ERROR loading custom dict: {e}")
                     raise ValueError("Failed to load custom vocabulary")
            else:
                raise ValueError("Vocabulary size not supported or CUSTOM_DICT not provided.")

                  
            assert len(self.vocab) == self.char_voc_size, f"Loaded vocab size {len(self.vocab)} != expected char_voc_size {self.char_voc_size}"
            # assert len(self.vocab) == self.voc_size, f"Loaded vocab size {len(self.vocab)} != config voc_size {self.voc_size}"

            # TODO: Define embedding dimension based on your chosen model (e.g., CANINE base is often 768)
            canine_model_name = "google/canine-c"
            embed_dim = 768 # Placeholder dimension
            #embeddings_dict, _ = load_char_embeddings(self.vocab, embed_dim) # Load embeddings
            #char_embeddings_dict = get_char_embeddings_canine(self.vocab, model_name=canine_model_name, device=device)
            char_embeddings_dict = get_char_embeddings_canine(cfg, self.vocab, device=device)
            
            # Load corpus
            # TODO: You might want to define the corpus source in config
            generic_dict_path = "datasets/generic_90k_words.txt"
            corpus = get_corpus(cfg.DATASETS.TRAIN, generic_dict_path)

            # Generate centroids - needs to run only once, ideally load precomputed
            # We do it here for simplicity, but moving outside Trainer init is better
            device = torch.device(cfg.MODEL.DEVICE)
            self.centroids = generate_centroids(embeddings_dict, corpus, self.vocab, device)
            global _canine_model_cache
            if canine_model_name in _canine_model_cache:
                 del _canine_model_cache[canine_model_name] # Remove model from cache
                 torch.cuda.empty_cache() # Try to clear GPU memory
            self.enc_matcher, self.dec_matcher = build_matcher(
            cfg,
            centroids=self.centroids,
            vocab=self.vocab
        )
    
    # Pass cfg to get temperature
    def generate_soft_distribution(cfg, char_centroid, all_centroids):
        temperature = cfg.MODEL.LINGUISTIC_PRIOR.TEMPERATURE # Read from cfg
        if char_centroid is None or torch.isnan(char_centroid).any() or (char_centroid == 0).all():
            warnings.warn("Character centroid invalid, zero, or missing, returning uniform distribution.")
            voc_size = all_centroids.shape[0]
            return torch.ones(voc_size, device=all_centroids.device) / voc_size
        similarities = F.cosine_similarity(char_centroid, all_centroids, dim=-1)
        distribution = F.softmax(similarities / temperature, dim=-1)
        return distribution

    def apply_distribution_threshold(cfg, distribution, target_idx, voc_size=-1):
        threshold = cfg.MODEL.LINGUISTIC_PRIOR.SOFT_TARGET_THRESHOLD # Read from cfg
    # ... (Rest of the logic from previous step) ...
        if voc_size == -1:
            voc_size = distribution.shape[0]
        if voc_size <= 1: # Handle edge case
            return distribution
        k = voc_size # Size of character vocabulary

        if distribution[target_idx] >= threshold:
            new_dist = torch.zeros_like(distribution)
            p1_val = (1.0 - threshold) / (k - 1)
            new_dist.fill_(p1_val)
            new_dist[target_idx] = threshold
            sum_val = new_dist.sum()
            if abs(sum_val - 1.0) > 1e-6: # Add tolerance for floating point
                new_dist /= sum_val
            return new_dist
        else:
            return torch.ones_like(distribution) / k

    def get_char_voc_size(self):
        return self.voc_size - 1

    def loss_labels(self, outputs, targets, indices, num_inst, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(src_logits.shape[:-1], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes_o = torch.cat([t["labels"][J]
                                      for t, (_, J) in zip(targets, indices)])
        if len(target_classes_o.shape) < len(target_classes[idx].shape):
            target_classes_o = target_classes_o[..., None]
        target_classes[idx] = target_classes_o

        shape = list(src_logits.shape)
        shape[-1] += 1
        target_classes_onehot = torch.zeros(shape,
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(-1, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[..., :-1]
        # src_logits, target_classes_onehot: (bs, nq, n_ctrl_pts, 1)
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_inst,
                                     alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_beziers(self, outputs, targets, indices, num_inst):
        # may FIX: (1) seg valid points
        assert 'pred_beziers' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_beziers = outputs['pred_beziers'][idx]
        src_beziers =  self.bezier_sampler.get_sample_points(src_beziers.view(-1, 4, 2))
        target_beziers = torch.cat(
            [t['beziers'][i] for t, (_, i) in zip(targets, indices)],
            dim=0
        )
        target_beziers = self.bezier_sampler.get_sample_points(target_beziers)
        if target_beziers.numel() == 0:
            target_beziers = src_beziers.clone().detach()
        loss_bezier = F.l1_loss(src_beziers, target_beziers, reduction='none')
        losses = {}
        losses['loss_bezier'] = loss_bezier.sum() / num_inst
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_inst):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.mean(-2).argmax(-1) == 0).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_texts(self, outputs, targets, indices, num_inst):
        # CTC loss for classification of points
        assert 'pred_text_logits' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_texts = outputs['pred_text_logits'][idx]  # shape: (n, length, voc_size+1)
        src_texts = src_texts.permute(1, 0, 2)
        src = F.log_softmax(src_texts, dim=-1)  # shape: (length, n, voc_size+1)

        target_texts = torch.cat([t['texts'][i] for t, (_, i) in zip(targets, indices)]) # n, length
        input_lengths = torch.full((src.size(1),), src.size(0), dtype=torch.long)
        target_lengths = (target_texts != self.voc_size).long().sum(dim=-1)
        target_texts = torch.cat([t[:l] for t, l in zip(target_texts, target_lengths)])

        return {
            'loss_texts': F.ctc_loss(
                src,
                target_texts,
                input_lengths,
                target_lengths,
                blank=self.voc_size,
                zero_infinity=True
            )
        }

    def loss_ctrl_points(self, outputs, targets, indices, num_inst):
        """Compute the L1 regression loss
        """
        assert 'pred_ctrl_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_ctrl_points = outputs['pred_ctrl_points'][idx]
        target_ctrl_points = torch.cat([t['ctrl_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_ctrl_points = F.l1_loss(src_ctrl_points, target_ctrl_points, reduction='sum')
        losses = {'loss_ctrl_points': loss_ctrl_points / num_inst}
        return losses

    def loss_bd_points(self, outputs, targets, indices, num_inst):
        assert 'pred_bd_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_bd_points = outputs['pred_bd_points'][idx]
        target_bd_points = torch.cat([t['bd_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bd_points = F.l1_loss(src_bd_points, target_bd_points, reduction='sum')
        losses = {'loss_bd_points': loss_bd_points / num_inst}
        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                               for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                               for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_inst, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'ctrl_points': self.loss_ctrl_points,
            'beziers': self.loss_beziers,
            'texts': self.loss_texts,
            'bd_points': self.loss_bd_points
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_inst, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.dec_matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_inst = sum(len(t['ctrl_points']) for t in targets)
        num_inst = torch.as_tensor(
            [num_inst], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_inst)
        num_inst = torch.clamp(num_inst / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        for loss in self.dec_losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_inst, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.dec_matcher(aux_outputs, targets)
                for loss in self.dec_losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_inst, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices = self.enc_matcher(enc_outputs, targets)
            for loss in self.enc_losses:
                kwargs = {}
                if loss == 'labels':
                    kwargs['log'] = False
                l_dict = self.get_loss(
                    loss, enc_outputs, targets, indices, num_inst, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses