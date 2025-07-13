from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def answer_query(query: str, chunks: list):
    chunk_embeddings = model.encode(chunks)
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    best_index = int(np.argmax(similarities))
    best_chunk = chunks[best_index]
    return best_chunk.strip(), f"Found in chunk #{best_index + 1}"
