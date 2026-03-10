from sentence_transformers import SentenceTransformer

# Free embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def get_embedding(text):

    embedding = model.encode(text, normalize_embeddings=True)

    return embedding