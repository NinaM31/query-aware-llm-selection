import numpy as np


def cosine_scores(query_embeddings, candidate_embeddings, clip_negative=True):
    '''
        Compute cosine similarity scores between query embeddings and candidate embeddings 
        Inputs must already be L2-normalized.
    '''
    scores = query_embeddings @ candidate_embeddings.T

    if clip_negative:
        scores = np.maximum(scores, 0)
    return scores.astype(np.float32)