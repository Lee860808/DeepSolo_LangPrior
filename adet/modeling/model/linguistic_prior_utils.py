# DeepSolo_LangPrior/DeepSolo/adet/modeling/model/linguistic_prior_utils.py
import torch
import torch.nn.functional as F
import numpy as np
import warnings
import json
import os
from tqdm import tqdm
from transformers import CanineTokenizer, CanineModel # Import CANINE specific tools
from detectron2.data import MetadataCatalog # To get annotation file paths

# --- CANINE Model Loading ---
# Global cache for the model and tokenizer to avoid reloading
_canine_model_cache = {}
_canine_tokenizer_cache = {}

def get_canine_model(model_name="google/canine-c", device='cpu'):
    """Loads CANINE model and tokenizer, using a cache."""
    global _canine_model_cache, _canine_tokenizer_cache
    if model_name not in _canine_model_cache:
        print(f"Loading CANINE model: {model_name}...")
        model = CanineModel.from_pretrained(model_name).to(device)
        model.eval() # Set to evaluation mode
        _canine_model_cache[model_name] = model
        print(f"Loading CANINE tokenizer: {model_name}...")
        tokenizer = CanineTokenizer.from_pretrained(model_name)
        _canine_tokenizer_cache[model_name] = tokenizer
    return _canine_model_cache[model_name], _canine_tokenizer_cache[model_name]

@torch.no_grad()
def get_char_embeddings_canine(characters, model_name="google/canine-c", device='cpu'):
    """
    Computes embeddings for a list of characters using CANINE.
    Args:
        characters (list): List of unique characters.
        model_name (str): Name of the CANINE model to use.
        device: Torch device.
    Returns:
        dict: Mapping character -> embedding tensor.
    """
    model, tokenizer = get_canine_model(model_name, device)
    embeddings_dict = {}
    print(f"Generating character embeddings using {model_name}...")

    # Process characters one by one (CANINE handles single characters)
    for char in tqdm(characters):
        # CANINE expects strings
        inputs = tokenizer(char, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        # CANINE output includes special tokens. We typically want the embedding
        # corresponding to the actual character token(s).
        # For single chars, it's usually the state after the initial special token.
        # Let's take the hidden state corresponding to the first actual character token ID.
        # Input IDs for 'a' might be [2, 97, 3] (special, 'a', special). We want index 1.
        # Find the first non-special token index
        token_ids = inputs['input_ids'][0].tolist()
        first_char_token_idx = -1
        for idx, token_id in enumerate(token_ids):
            # Assuming special tokens have low IDs (e.g., < 4 for CANINE's standard special tokens)
            # Or check against tokenizer.all_special_ids
            if token_id not in tokenizer.all_special_ids:
                 first_char_token_idx = idx
                 break

        if first_char_token_idx != -1:
            # Extract the embedding for the first character token
            char_embedding = outputs.last_hidden_state[0, first_char_token_idx, :].detach().cpu()
            embeddings_dict[char] = char_embedding
        else:
            warnings.warn(f"Could not find valid character token for '{char}' in CANINE output. Skipping.")
            embeddings_dict[char] = None # Indicate failure

    print("Character embedding generation complete.")
    return embeddings_dict
def load_char_embeddings(vocab, embed_dim):
    """
    Loads or computes character embeddings.
    Args:
        vocab (list): List of characters in the vocabulary (excluding blank/unknown).
        embed_dim (int): Embedding dimension.
    Returns:
        dict: Mapping character -> embedding tensor.
        torch.Tensor: Centroid matrix (voc_size, embed_dim). Placeholder for now.
    """
    print("WARNING: Using placeholder character embeddings and centroids!")
    # TODO: Implement actual character embedding loading (e.g., using CANINE)
    # For now, return random embeddings and centroids
    embeddings = {char: torch.randn(embed_dim) for char in vocab}
    centroids = torch.randn(len(vocab), embed_dim) # voc_size x embed_dim
    return embeddings, centroids

# --- Corpus Loading ---
def get_corpus(dataset_names, generic_dict_path="datasets/generic_90k_words.txt"):
    """
    Loads words from training datasets annotations and an optional generic dictionary.
    Args:
        dataset_names (tuple): Names of training datasets registered in Detectron2.
        generic_dict_path (str): Path to the generic word list file.
    Returns:
        list: A list of unique words making up the corpus (converted to uppercase).
    """
    print("Building corpus...")
    corpus_words = set()

    # 1. Load from dataset annotations
    for dataset_name in dataset_names:
        try:
            metadata = MetadataCatalog.get(dataset_name)
            json_file = metadata.json_file
            print(f"  Loading annotations from: {json_file}")
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'annotations' in data: # Standard COCO-like format
                    for ann in data['annotations']:
                        # Look for recognition transcription ('rec' commonly used in text datasets)
                        if 'rec' in ann:
                            # 'rec' might be list of ints or string, handle both
                            rec_data = ann['rec']
                            if isinstance(rec_data, list): # Skip if it's indices directly
                                print(f"    Skipping annotation with list 'rec' in {dataset_name}")
                                continue
                            elif isinstance(rec_data, str):
                                word = rec_data.strip()
                                if word:
                                    corpus_words.add(word.upper())
                        elif 'utf8_string' in ann: # Another common key
                            word = ann['utf8_string'].strip()
                            if word:
                                 corpus_words.add(word.upper())
                        # Add other potential keys for transcription if needed
                else:
                     print(f"    Warning: No 'annotations' key found in {json_file}")
        except Exception as e:
            print(f"  Warning: Could not load or parse annotations for {dataset_name}. Error: {e}")

    # 2. Load from generic dictionary
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

# --- Centroid Generation (Modified) ---
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
    # Infer embed_dim, find first valid embedding
    embed_dim = 0
    for emb in embeddings_dict.values():
        if emb is not None:
            embed_dim = emb.shape[0]
            break
    if embed_dim == 0:
        raise ValueError("Could not determine embedding dimension from embeddings_dict.")

    # Initialize centroids and counts on the target device
    centroids = torch.zeros(voc_size, embed_dim, device=device)
    counts = torch.zeros(voc_size, device=device)
    char_to_idx = {char: idx for idx, char in enumerate(vocab)} # lowercase vocab -> index

    processed_chars = 0
    for word in tqdm(corpus): # Corpus is already uppercase
        for char in word:
            char_lower = char.lower() # Convert to lowercase for vocab matching
            if char_lower in char_to_idx:
                idx = char_to_idx[char_lower]
                # Use the lowercase char to find the embedding
                if char_lower in embeddings_dict and embeddings_dict[char_lower] is not None:
                    # Move embedding to target device right before adding
                    centroids[idx] += embeddings_dict[char_lower].to(device)
                    counts[idx] += 1
                    processed_chars += 1
                # else: # Already handled by warning in embedding generation
                     # pass

    print(f"Processed {processed_chars} character instances from corpus.")

    # Average embeddings
    valid_counts = counts > 0
    counts_clamped = counts.clamp(min=1) # Avoid division by zero
    centroids /= counts_clamped.unsqueeze(-1)

    num_found = int(valid_counts.sum())
    if num_found < voc_size:
        warnings.warn(f"Centroids generated. Found {num_found}/{voc_size} characters in corpus. "
                      f"Characters without corpus instances will have zero vectors (or initial random state if provided).")
        # Optional: Fill missing centroids with average or random noise?
        # avg_centroid = centroids[valid_counts].mean(dim=0)
        # zero_centroid_indices = ~valid_counts
        # if zero_centroid_indices.any():
        #     centroids[zero_centroid_indices] = avg_centroid

    print("Centroid generation complete.")
    return centroids
def get_corpus(dataset_names):
    """
    Loads words from training datasets and/or a generic dictionary.
    Args:
        dataset_names (tuple): Names of training datasets registered in Detectron2.
    Returns:
        list: A list of words making up the corpus.
    """
    print("WARNING: Using placeholder corpus!")
    # TODO: Implement corpus loading from training annotations and/or generic dict
    # Example: Load words from TotalText annotations and a generic dictionary
    corpus = ["HELLO", "WORLD", "TOTAL", "TEXT", "SCENE", "LANGUAGE", "PRIOR", "DEEP", "SOLO"] # Placeholder
    # Add words from dataset annotations here if desired
    return corpus

def generate_centroids(embeddings_dict, corpus, vocab, device):
    """
    Generates centroids by averaging embeddings for characters in the corpus.
    Args:
        embeddings_dict (dict): char -> embedding tensor.
        corpus (list): List of words.
        vocab (list): List of characters in the vocabulary.
        device: Torch device.
    Returns:
        torch.Tensor: Centroid matrix (voc_size, embed_dim).
    """
    print("Generating centroids...")
    voc_size = len(vocab)
    embed_dim = next(iter(embeddings_dict.values())).shape[0]
    centroids = torch.zeros(voc_size, embed_dim, device=device)
    counts = torch.zeros(voc_size, device=device)
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}

    for word in corpus:
        for char in word:
            char_lower = char.lower() # Assuming case-insensitivity matching vocab
            if char_lower in char_to_idx:
                idx = char_to_idx[char_lower]
                # TODO: Ensure embedding_dict keys match how you process chars (e.g., case)
                if char_lower in embeddings_dict:
                    centroids[idx] += embeddings_dict[char_lower].to(device)
                    counts[idx] += 1
                # else:
                    # print(f"Warning: Character '{char}' not in embeddings_dict")

    # Average embeddings
    counts[counts == 0] = 1 # Avoid division by zero
    centroids /= counts.unsqueeze(-1)

    # Handle characters not seen in corpus (optional: use initial random or zero vector?)
    # For now, they remain zero or their initial random state if load_char_embeddings provides them
    print(f"Centroid generation complete. Found {int(sum(counts > 1))} characters in corpus.")
    return centroids

def generate_soft_distribution(char_centroid, all_centroids, temperature=1.0):
    """
    Generates the soft probability distribution for a single character based on centroid similarity.
    Args:
        char_centroid (Tensor): Embedding centroid for the target character (1, embed_dim).
        all_centroids (Tensor): Matrix of all character centroids (voc_size, embed_dim).
        temperature (float): Softmax temperature.
    Returns:
        Tensor: Soft probability distribution (voc_size,).
    """
    if char_centroid is None or torch.isnan(char_centroid).any():
         # Handle case where centroid might not exist (not in corpus)
         # Return a uniform distribution or some default
         warnings.warn("Character centroid invalid or missing, returning uniform distribution.")
         voc_size = all_centroids.shape[0]
         return torch.ones(voc_size, device=all_centroids.device) / voc_size

    # Cosine similarity or Dot product? Paper implies dot product in Eq 2's exp term.
    similarities = F.cosine_similarity(char_centroid, all_centroids, dim=-1) # Shape: (voc_size,)
    # similarities = torch.matmul(char_centroid, all_centroids.t()).squeeze(0) # Shape: (voc_size,)

    distribution = F.softmax(similarities / temperature, dim=-1)
    return distribution

def apply_distribution_threshold(distribution, target_idx, threshold=0.85, voc_size=-1):
    """
    Applies the thresholding described in LangPrior Eq 6.
    Args:
        distribution (Tensor): The initial soft distribution (voc_size,).
        target_idx (int): The index of the actual target character.
        threshold (float): The probability threshold T.
        voc_size (int): Size of the vocabulary (excluding blank/unknown). If -1, inferred.
    Returns:
        Tensor: The post-processed distribution (voc_size,).
    """
    if voc_size == -1:
        voc_size = distribution.shape[0]
    k = voc_size # Size of character vocabulary
    if distribution[target_idx] >= threshold:
        # Keep the probability at target_idx
        new_dist = torch.zeros_like(distribution)
        p1_val = (1.0 - threshold) / (k - 1) if k > 1 else 0.0 # Value for other chars
        new_dist.fill_(p1_val)
        new_dist[target_idx] = threshold
        # Renormalize slightly if needed due to float precision
        new_dist /= new_dist.sum()
        return new_dist
    else:
        # Use a predefined uniform-like distribution if threshold not met
        # Or maybe just return the original? Paper says "a predefined distribution is used"
        # Let's return a uniform one for now.
        warnings.warn(f"Threshold not met for char idx {target_idx}. Returning uniform distribution.")
        return torch.ones_like(distribution) / k

def generate_soft_target_sequence(text_indices, text_lengths, centroids, vocab, threshold=0.85):
    """
    Generates a sequence of soft distributions for a batch of ground truth texts.
    Args:
        text_indices (Tensor): Ground truth character indices (total_gt_chars,)
        text_lengths (Tensor): Lengths of each text instance (num_gt,)
        centroids (Tensor): Centroid matrix (voc_size, embed_dim).
        vocab (list): List of characters.
        threshold (float): Probability threshold T for post-processing.
    Returns:
        list[Tensor]: A list where each element is the soft target sequence (L_i, voc_size)
                      for the i-th ground truth text.
    """
    voc_size = len(vocab)
    target_sequences = []
    current_idx = 0
    char_to_idx = {char: idx for idx, char in enumerate(vocab)} # Assuming vocab matches centroid rows

    for length in text_lengths:
        if length == 0:
             target_sequences.append(torch.empty(0, voc_size, device=centroids.device))
             continue

        indices_for_seq = text_indices[current_idx : current_idx + length]
        soft_targets_for_seq = []
        for char_idx in indices_for_seq:
            if char_idx >= voc_size or char_idx < 0: # Handle potential padding or invalid indices
                # Append a uniform distribution? Or skip? Let's use uniform.
                dist = torch.ones(voc_size, device=centroids.device) / voc_size
            else:
                char_centroid = centroids[char_idx].unsqueeze(0) # (1, embed_dim)
                dist = generate_soft_distribution(char_centroid, centroids)
                # Apply thresholding based on the original character index
                dist = apply_distribution_threshold(dist, char_idx, threshold, voc_size)

            soft_targets_for_seq.append(dist)

        target_sequences.append(torch.stack(soft_targets_for_seq, dim=0)) # (L_i, voc_size)
        current_idx += length

    return target_sequences # List of tensors with shape (L_i, voc_size)