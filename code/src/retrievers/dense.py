import numpy as np
from src.scoring.cosine import cosine_scores

class DenseRetriever:
    '''Retrieve candidates using cosine similarity between query embeddings and candidate embeddings'''
    def __init__(self, candidate_ids, candidate_embeddings):
        self.candidate_ids = candidate_ids
        self.candidate_embeddings = candidate_embeddings

    def search(self, query_embedding, top_k):
        '''Rank candidate LLMs for one query'''
        # Add query dimension:
        # (384,) -> (1, 384)
        query_embedding = query_embedding.reshape(1, -1)

        # Cosine similarity against all candidate vectors
        scores = cosine_scores(query_embedding, self.candidate_embeddings)[0]

        # Rank highest score first
        order = np.argsort(-scores, kind='stable')[:top_k]
        results = []

        for rank, idx in enumerate(order, start=1):
            results.append({
                'rank': rank,
                'llm_id': self.candidate_ids[idx],
                'score': float(scores[idx])
            })

        return results