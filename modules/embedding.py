# modules/embedding.py
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from config import SBERT_MODEL_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(SBERT_MODEL_NAME).to(device)

def get_embedding(text):
    return model.encode(text, convert_to_tensor=True).to(device)

def compute_semantic_similarity(text1, text2):
    embedding1 = model.encode(text1, convert_to_tensor=True).to(device)
    embedding2 = model.encode(text2, convert_to_tensor=True).to(device)
    from sentence_transformers import util
    return util.pytorch_cos_sim(embedding1, embedding2).item()

def compute_tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
