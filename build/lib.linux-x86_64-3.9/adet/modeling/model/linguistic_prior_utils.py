# DeepSolo_LangPrior/DeepSolo/adet/modeling/model/linguistic_prior_utils.py
import torch
import torch.nn.functional as F
import numpy as np
import warnings
import json
import os
import pickle
from tqdm import tqdm
from transformers import CanineTokenizer, CanineModel # Import CANINE specific tools
from detectron2.data import MetadataCatalog # To get annotation file paths
from detectron2.utils.file_io import PathManager # Use D2's file handling

# --- CANINE Model Loading ---
# Global cache for the model and tokenizer to avoid reloading
_canine_model_cache = {}
_canine_tokenizer_cache = {}

def get_canine_model(cfg, device='cpu'):
    """Loads CANINE model and tokenizer, using a cache."""
    global _canine_model_cache, _canine_tokenizer_cache
    model_name = cfg.MODEL.LINGUISTIC_PRIOR.EMBED_MODEL_NAME
    cache_key = f"{model_name}_{str(device)}"
    if cache_key not in _canine_model_cache:
        print(f"Loading CANINE model: {model_name} onto {device}...")
        try:
            model = CanineModel.from_pretrained(model_name).to(device)
            model.eval()
            _canine_model_cache[cache_key] = model
            print(f"Loading CANINE tokenizer: {model_name}...")
            tokenizer = CanineTokenizer.from_pretrained(model_name)
            _canine_tokenizer_cache[model_name] = tokenizer
        except Exception as e:
            print(f"ERROR loading CANINE model/tokenizer ({model_name}): {e}")
            raise e
    return _canine_model_cache[cache_key], _canine_tokenizer_cache[model_name]

@torch.no_grad()
def get_char_embeddings_canine(cfg, characters, device='cpu'):
    """
    Computes embeddings for a list of characters using CANINE.
    (Revised V5: Pad short inputs before tokenization)
    Args:
        cfg (CfgNode): Configuration object.
        characters (list): List of unique characters.
        device: Torch device for CANINE model execution.
    Returns:
        dict: Mapping character -> embedding tensor (on CPU).
    """
    model_name = cfg.MODEL.LINGUISTIC_PRIOR.EMBED_MODEL_NAME
    model, tokenizer = get_canine_model(cfg, device)
    embedding_layer = model.embeddings # Use embedding layer directly as in V3

    # --- Define Padding Strategy ---
    # We need the sequence length AFTER tokenization to be >= 4
    # Tokenizer often adds [CLS] and [SEP] (2 tokens)
    # So, the character(s) themselves need to contribute >= 2 tokens.
    # Single common chars usually map to 1 token ID. Let's pad to be safe.
    min_char_len_for_tokenizer = 3 # Pad single chars to length 3, e.g., "a__"
    pad_char = "_" # Choose a padding char unlikely to be OOV or special

    embeddings_dict = {}
    print(f"Generating character embeddings using {model_name} (Pad Input V5)...")

    failed_chars = []
    for char in tqdm(characters):
        if not char or char.isspace():
            embeddings_dict[char] = None
            failed_chars.append(repr(char))
            continue

        try:
            # --- Pad the input character string ---
            padded_char_str = char
            if len(char) < min_char_len_for_tokenizer:
                 padded_char_str = char.ljust(min_char_len_for_tokenizer, pad_char)
            # ------------------------------------

            # 1. Tokenize the (potentially padded) string
            inputs = tokenizer(padded_char_str, return_tensors="pt", padding=False, truncation=False).to(device)
            input_ids = inputs['input_ids']

            # Defensive check for empty tokenization
            if input_ids.shape[1] == 0:
                 warnings.warn(f"Tokenizer returned empty sequence for padded '{repr(padded_char_str)}' (original '{repr(char)}'). Skipping.")
                 failed_chars.append(repr(char))
                 embeddings_dict[char] = None
                 continue

            # 2. Pass *only input_ids* to the embedding layer's forward method
            embedding_output = embedding_layer(input_ids=input_ids)

            # 3. Find the correct index for the *original* character
            token_ids_list = input_ids[0].tolist()
            first_char_token_idx = -1
            # Re-tokenize the *original* char to know its expected token ID(s)
            original_tokens = tokenizer.encode(char, add_special_tokens=False)
            if not original_tokens:
                 warnings.warn(f"Original char '{repr(char)}' tokenized to empty list. Skipping.")
                 failed_chars.append(repr(char))
                 embeddings_dict[char] = None
                 continue
            first_original_token_id = original_tokens[0]

            # Find where this token ID appears after the initial special tokens in the padded sequence
            found_idx = -1
            for idx, token_id in enumerate(token_ids_list):
                 if token_id not in tokenizer.all_special_ids:
                     if token_id == first_original_token_id:
                          found_idx = idx
                          break # Found the first instance of the original char's token
                     # elif found_idx == -1: # Keep track of first non-special if exact match fails? Maybe too complex.
                     #    found_idx = idx

            first_char_token_idx = found_idx # Use the found index


            if first_char_token_idx != -1 and first_char_token_idx < embedding_output.shape[1]:
                final_embedding = embedding_output[0, first_char_token_idx, :].detach().cpu()
                embeddings_dict[char] = final_embedding
            else:
                warnings.warn(f"Could not find valid token index {first_char_token_idx} for original char '{repr(char)}' in padded sequence tokens: {token_ids_list}. Skipping.")
                failed_chars.append(repr(char))
                embeddings_dict[char] = None

        except Exception as e:
             warnings.warn(f"Error processing character '{repr(char)}' with CANINE: {e}")
             failed_chars.append(repr(char))
             embeddings_dict[char] = None

    if failed_chars:
        warnings.warn(f"Could not generate embeddings for {len(failed_chars)} characters: {failed_chars}")
    print("Character embedding generation complete.")
    return embeddings_dict

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
    generic_dict_path = cfg.MODEL.LINGUISTIC_PRIOR.GENERIC_DICT_PATH

    # 1. Load from dataset annotations
    for dataset_name in dataset_names:
        try:
            metadata = MetadataCatalog.get(dataset_name)
            json_file = PathManager.get_local_path(metadata.json_file)
            print(f"  Loading annotations from: {json_file}")
            with PathManager.open(json_file, 'r') as f:
                data = json.load(f)
                if 'annotations' in data:
                    for ann in data['annotations']:
                        text = None
                        # Adjust keys based on common text annotation formats
                        if 'text' in ann and isinstance(ann['text'], str):
                            text = ann['text']
                        elif 'rec' in ann and isinstance(ann['rec'], str):
                             text = ann['rec']
                        elif 'utf8_string' in ann and isinstance(ann['utf8_string'], str):
                             text = ann['utf8_string']

                        if text:
                             word = text.strip()
                             # Basic filtering can be added here if needed
                             if word and len(word) > 0: # Keep even single characters
                                 corpus_words.add(word.upper())

                else:
                     print(f"    Warning: No 'annotations' key found in {json_file}")
        except FileNotFoundError:
             print(f"  Warning: Annotation file not found: {json_file}. Skipping {dataset_name}.")
        except KeyError:
             print(f"  Warning: Metadata for dataset '{dataset_name}' not found. Skipping.")
        except Exception as e:
            print(f"  Warning: Could not load or parse annotations for {dataset_name}. Error: {e}")


    # 2. Load from generic dictionary
    if generic_dict_path:
        local_dict_path = PathManager.get_local_path(generic_dict_path)
        if PathManager.exists(local_dict_path):
            print(f"  Loading generic dictionary from: {local_dict_path}")
            try:
                with PathManager.open(local_dict_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            corpus_words.add(word.upper())
            except Exception as e:
                print(f"  Warning: Could not load generic dictionary '{local_dict_path}'. Error: {e}")
        else:
            print(f"  Warning: Generic dictionary path not found: {local_dict_path}")
    else:
        print("  Warning: No generic dictionary path specified.")

    corpus_list = sorted(list(corpus_words))
    print(f"Corpus built with {len(corpus_list)} unique words.")
    if not corpus_list:
        warnings.warn("Corpus is empty! Centroid calculation will produce zero vectors.")
    return corpus_list

# --- Centroid Generation (Modified for robustness) ---
def generate_centroids(embeddings_dict, corpus, vocab, device):
    """
    Generates centroids by averaging embeddings for characters in the corpus.
    Args:
        embeddings_dict (dict): char -> embedding tensor (on CPU).
        corpus (list): List of uppercase words.
        vocab (list): List of characters (lowercase).
        device: Torch device.
    Returns:
        torch.Tensor: Centroid matrix (voc_size, embed_dim) on the specified device.
    """
    print("Generating centroids...")
    voc_size = len(vocab)
    char_to_idx = {char: idx for idx, char in enumerate(vocab)} # lowercase vocab -> index

    # Infer embed_dim robustly
    embed_dim = 0
    valid_embed_found = False
    for emb in embeddings_dict.values():
        if emb is not None:
            embed_dim = emb.shape[0]
            valid_embed_found = True
            break
    if not valid_embed_found:
        # Handle case where NO embeddings were generated
        warnings.warn("No valid character embeddings found in embeddings_dict. Cannot generate centroids.")
        return torch.zeros(voc_size, 768, device=device) # Return zeros with default CANINE dim

    centroids = torch.zeros(voc_size, embed_dim, device=device, dtype=torch.float32) # Use float32
    counts = torch.zeros(voc_size, device=device, dtype=torch.float32)

    processed_chars = 0
    for word in tqdm(corpus): # Corpus is already uppercase
        for char in word:
            char_lower = char.lower() # Convert to lowercase for vocab matching
            if char_lower in char_to_idx:
                idx = char_to_idx[char_lower]
                if char_lower in embeddings_dict and embeddings_dict[char_lower] is not None:
                    centroids[idx] += embeddings_dict[char_lower].to(device, dtype=torch.float32)
                    counts[idx] += 1.0
                    processed_chars += 1

    print(f"Processed {processed_chars} character instances from corpus.")

    # Average embeddings, handle division by zero
    valid_counts_mask = counts > 0
    centroids[valid_counts_mask] /= counts[valid_counts_mask].unsqueeze(-1)

    num_found = int(valid_counts_mask.sum())
    if num_found < voc_size:
        missing_chars = [v for v_idx, v in enumerate(vocab) if not valid_counts_mask[v_idx].item()]
        warnings.warn(f"Centroids generated. Found {num_found}/{voc_size} characters in corpus. "
                      f"Characters without corpus instances have zero vectors: {missing_chars}")
        # Consider alternative strategies for missing chars if needed

    print("Centroid generation complete.")
    return centroids

# Pass cfg
def generate_soft_distribution(cfg, char_centroid, all_centroids):
    """
    Generates the soft probability distribution for a single character based on centroid similarity.
    """
    temperature = cfg.MODEL.LINGUISTIC_PRIOR.TEMPERATURE
    if char_centroid is None or (not torch.is_tensor(char_centroid)) or torch.isnan(char_centroid).any() or (char_centroid == 0).all():
         warnings.warn("Character centroid invalid, zero, or missing, returning uniform distribution.")
         voc_size = all_centroids.shape[0]
         return torch.ones(voc_size, device=all_centroids.device) / voc_size

    # Ensure dimensions match for similarity calculation
    if char_centroid.ndim == 1:
        char_centroid = char_centroid.unsqueeze(0) # Make it (1, embed_dim)

    similarities = F.cosine_similarity(char_centroid, all_centroids, dim=-1) # Shape: (voc_size,)
    distribution = F.softmax(similarities / temperature, dim=-1)
    return distribution

# Pass cfg
def apply_distribution_threshold(cfg, distribution, target_idx, voc_size=-1):
    """
    Applies the thresholding described in LangPrior Eq 6.
    """
    threshold = cfg.MODEL.LINGUISTIC_PRIOR.SOFT_TARGET_THRESHOLD
    if voc_size == -1:
        voc_size = distribution.shape[0]
    if voc_size <= 1:
        return distribution # Avoid division by zero if voc_size is 1
    k = voc_size

    # Ensure target_idx is valid
    if not (0 <= target_idx < k):
        warnings.warn(f"Invalid target_idx {target_idx} for vocab size {k}. Returning uniform distribution.")
        return torch.ones_like(distribution) / k

    if distribution[target_idx] >= threshold:
        new_dist = torch.zeros_like(distribution)
        p1_val = (1.0 - threshold) / (k - 1)
        new_dist.fill_(p1_val)
        new_dist[target_idx] = threshold
        sum_val = new_dist.sum()
        if abs(sum_val - 1.0) > 1e-6:
             new_dist /= sum_val # Normalize
        return new_dist
    else:
        # Paper doesn't specify, let's return the original distribution in this case
        # instead of uniform, potentially retaining some info.
        # warnings.warn(f"Threshold not met for char idx {target_idx} (P={distribution[target_idx]:.4f} < {threshold}). Returning original distribution.")
        # return distribution
        # Or, stick to uniform as per previous implementation:
        return torch.ones_like(distribution) / k

# Pass cfg
def generate_soft_target_sequence(cfg, text_indices, text_lengths, centroids, vocab):
    voc_size = len(vocab)
    target_sequences = []
    current_idx = 0
    
    for length in text_lengths:
        length_int = int(length.item()) # Ensure length is an integer
        if length_int <= 0:
             target_sequences.append(torch.empty(0, voc_size, device=centroids.device))
             continue

        indices_for_seq = text_indices[current_idx : current_idx + length_int]
        soft_targets_for_seq = []
        for char_idx in indices_for_seq:
            char_idx_int = int(char_idx.item())
            if not (0 <= char_idx_int < voc_size): # Check index validity
                dist = torch.ones(voc_size, device=centroids.device) / voc_size
            else:
                char_centroid = centroids[char_idx_int] # Already unsqueezed in generate_centroids
                dist = generate_soft_distribution(cfg, char_centroid, centroids)
                dist = apply_distribution_threshold(cfg, dist, char_idx_int, voc_size)

            soft_targets_for_seq.append(dist)

        if soft_targets_for_seq:
            target_sequences.append(torch.stack(soft_targets_for_seq, dim=0))
        else:
            target_sequences.append(torch.empty(0, voc_size, device=centroids.device))

        current_idx += length_int

    return target_sequences