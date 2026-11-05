import numpy as np
from sklearn.cluster import KMeans

def hierarchical_scores(
    query_embeddings,
    interaction_embeddings,
    interactions,
    llm_ids,
    centroids,
    n_local=10,
    broad_weight=0.8,
    local_weight=0.2,
    clip_negative=True
):
    '''
        Compute Hierarchical scores for each query and LLM.

        For each LLM:
            broad = best similarity to its centroids
            local = mean similarity of its top historical interactions
            final = broad_weight * broad + local_weight * local
        
        Output:
            numpy array -> shape (n_queries, n_llms)
    '''
    scores = np.zeros((len(query_embeddings), len(llm_ids)), dtype=np.float32)

    # Precompute which interactions belong to each LLM
    llm_interaction_indices = {
        llm_id: np.flatnonzero(interactions['llm_id'].to_numpy() == llm_id)
        for llm_id in llm_ids
    }

    for query_idx, query_embedding in enumerate(query_embeddings):
        # Similarity between query and every historical interaction
        all_similarities = interaction_embeddings @ query_embedding

        for llm_idx, llm_id in enumerate(llm_ids):
            centroid_scores, top_local = get_hierarchical_evidence(
                query_embedding=query_embedding,
                all_interaction_similarities=all_similarities,
                interaction_indices=llm_interaction_indices[llm_id],
                llm_centroids=centroids[llm_idx],
                n_local=n_local,
                clip_negative=clip_negative
            )

            broad = float(centroid_scores.max())
            local = float(top_local.mean())

            scores[query_idx, llm_idx] = broad_weight * broad + local_weight * local

    return scores

def get_hierarchical_evidence(query_embedding, all_interaction_similarities, interaction_indices, llm_centroids, n_local=10, clip_negative=True):
    '''Get centroid and local interaction evidence for one LLM'''
    # Centroid evidence
    centroid_scores = llm_centroids @ query_embedding

    # Local historical interaction evidence
    local_scores = all_interaction_similarities[interaction_indices]

    if clip_negative:
        centroid_scores = np.maximum(centroid_scores, 0)
        local_scores = np.maximum(local_scores, 0)

    # Keep strongest local interactions
    k = min(n_local, len(local_scores))

    top_local = np.partition(local_scores, len(local_scores) - k)[-k:]

    # Keep deterministic order
    top_local = np.sort(top_local)[::-1]

    return (centroid_scores.astype(np.float32), top_local.astype(np.float32))

def build_kmeans_centroids(interactions, embeddings, n_centroids=10, seed=42, n_init=10):
    '''Build KMeans centroids for each LLM'''
    llm_ids = sorted(interactions['llm_id'].unique())

    embedding_dim = embeddings.shape[1]
    centroids = np.zeros((len(llm_ids), n_centroids, embedding_dim), dtype=np.float32)

    for llm_idx, llm_id in enumerate(llm_ids):
        indices = np.flatnonzero(interactions['llm_id'].to_numpy() == llm_id)

        llm_embeddings = embeddings[indices]
        
        # Some LLMs may have fewer historical interactions
        # than the requested number of centroids.
        effective_centroids = min(n_centroids, len(llm_embeddings))
        
        kmeans = KMeans(n_clusters=effective_centroids, random_state=seed, n_init=n_init).fit(llm_embeddings)
        
        llm_centroids = kmeans.cluster_centers_.astype(np.float32)
        
        # Normalize real centroids
        norms = np.linalg.norm(llm_centroids, axis=1, keepdims=True)
        
        llm_centroids /= np.maximum(norms, 1e-12)
        
        # Keep a fixed shape of (n_centroids, embedding_dim)
        # so the rest of H does not need to change.
        if effective_centroids < n_centroids:
            padding = np.zeros((n_centroids - effective_centroids, llm_embeddings.shape[1]), dtype=np.float32)
            llm_centroids = np.vstack([llm_centroids, padding])
            
        centroids[llm_idx] = llm_centroids
    return llm_ids, centroids