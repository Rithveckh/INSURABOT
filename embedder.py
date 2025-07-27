from sentence_transformers import SentenceTransformer

# Add trust_remote_code=True to avoid error
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

def embed_text(text):
    embedding = model.encode(text)
    return embedding.tolist()
