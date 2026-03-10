import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):

        self.index = faiss.IndexFlatL2(dimension)
        self.text_chunks = []
        self.metadata = []

    def add_embeddings(self, embeddings, texts, pages):

        vectors = np.array(embeddings).astype("float32")

        self.index.add(vectors)

        self.text_chunks.extend(texts)
        self.metadata.extend(pages)

    def search(self, query_vector, k=20):

        query_vector = np.array([query_vector]).astype("float32")

        distances, indices = self.index.search(query_vector, k)

        results = []

        for idx in indices[0]:
            if idx < len(self.text_chunks):
                results.append({
                    "text": self.text_chunks[idx],
                    "page": self.metadata[idx]
                })

        return results