import numpy as np

def response_combsum_scores(query_embeddings, interaction_embeddings, interaction_llm_ids, llm_ids, top_k=1000, clip_negative=True):
    '''
        Compute Response-based CombSUM scores
        For each query:
            1. Compare against all historical interactions
            2. Keep the global top-k interactions
            3. Clip negative similarities to zero
            4. Sum retrieved scores by owning LLM

        Output:
            numpy array -> shape (n_queries, n_llms)
    '''
    llm_to_idx = {llm_id: i for i, llm_id in enumerate(llm_ids)}
    interaction_owner_indices = np.array([llm_to_idx[llm_id] for llm_id in interaction_llm_ids])

    scores = np.zeros((len(query_embeddings), len(llm_ids)), dtype=np.float32)

    for query_idx, query_embedding in enumerate(query_embeddings):
        top_scores, owners = get_response_evidence(
            query_embedding=query_embedding,
            interaction_embeddings=interaction_embeddings,
            interaction_owner_indices=interaction_owner_indices,
            top_k=top_k,
            clip_negative=clip_negative
        )
        # CombSUM
        np.add.at(scores[query_idx], owners, top_scores)
    return scores

def get_response_evidence(query_embedding, interaction_embeddings, interaction_owner_indices, top_k=1000, clip_negative=True):
    '''Retrieve the strongest historical interaction evidence for one query'''
    # Similarity to every historical interaction
    similarities = interaction_embeddings @ query_embedding

    # Keep global top-k interactions
    k = min(top_k, len(similarities))
    top_indices = np.argpartition(-similarities, k - 1)[:k]

    # Clip negative similarities
    scores = similarities[top_indices]
    if clip_negative: scores = np.maximum(scores, 0)
    scores = scores.astype(np.float32)

    # Find which LLM produced each retrieved interaction
    owners = interaction_owner_indices[top_indices]
    return scores, owners
