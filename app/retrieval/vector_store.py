import faiss
import numpy as np


class VectorStore:
    """
    FAISS-based vector store for semantic similarity search.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

        self.index = faiss.IndexFlatIP(dimension)

        self.documents = []

    def add(self, embeddings, documents: list[dict]):
        """
        Add embeddings and their corresponding document chunks.
        """

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        self.index.add(embeddings)

        self.documents.extend(documents)

    def search(
        self,
        query_embedding,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find the most semantically similar document chunks.
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        ).reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            document = self.documents[index].copy()

            document["score"] = float(score)

            results.append(document)

        return results