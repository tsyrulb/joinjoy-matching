from sentence_transformers import SentenceTransformer, util
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)

def get_embedding(text):
    return model.encode(text, convert_to_tensor=True).to(device)

def load_key_value_pairs(file_path):
    key_value_pairs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            key, value = line.strip().split(' ', 1)
            key_value_pairs.append((key, value))
    return key_value_pairs

file_path = 'wikidata_p1282_key_value_cleaned.txt'
key_value_pairs = load_key_value_pairs(file_path)

# Precompute embeddings for all keys and values once
keys = [kv[0] for kv in key_value_pairs]
values = [kv[1] for kv in key_value_pairs]

key_embeddings = model.encode(keys, convert_to_tensor=True, device=device)
value_embeddings = model.encode(values, convert_to_tensor=True, device=device)
# key_embeddings shape: [N, dim]
# value_embeddings shape: [N, dim]

def find_top_matches(user_input, key_value_pairs, top_n=20, key_weight=0.3, value_weight=0.7):
    # Compute user embedding once
    user_embedding = get_embedding(user_input)  # shape [dim]

    # Reshape for broadcasting
    user_embedding_expanded = user_embedding.unsqueeze(0)  # shape: [1, dim]

    # Compute cosine similarity with all keys and values in one go
    # cos_sim(user_embedding_expanded, key_embeddings) returns [1, N]
    key_sims = util.cos_sim(user_embedding_expanded, key_embeddings)[0]    # shape: [N]
    value_sims = util.cos_sim(user_embedding_expanded, value_embeddings)[0] # shape: [N]

    # Compute weighted similarities
    weighted_sims = key_weight * key_sims + value_weight * value_sims

    # Get top N indices
    top_indices = torch.argsort(weighted_sims, descending=True)[:top_n]

    results = []
    for idx in top_indices:
        sim = weighted_sims[idx].item()
        k, v = key_value_pairs[idx]
        results.append((sim, k, v))
    return results
