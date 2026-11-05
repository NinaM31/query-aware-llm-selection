from src.representations.profile import build_profile_representation, build_mean_profile_vectors
from src.representations.response import response_combsum_scores
from src.representations.hierarchical import hierarchical_scores
from src.cache.cache import load_or_create_embeddings, load_or_create_centroids

def compute_source_scores(interactions, interaction_embeddings, query_embeddings, config, sources):
    '''Compute requested P/R/H raw score matrices'''
    llm_ids = sorted(interactions['llm_id'].unique())
    source_scores = {}

    # PROFILE
    if 'P' in sources:
        _, profile_vectors = build_mean_profile_vectors(
            interactions,
            interaction_embeddings
        )

        source_scores['P'] = cosine_scores(
            query_embeddings,
            profile_vectors,
            clip_negative=config['clip_negative']
        )

    # RESPONSE
    if 'R' in sources:
        source_scores['R'] = response_combsum_scores(
            query_embeddings=query_embeddings,
            interaction_embeddings=interaction_embeddings,
            interaction_llm_ids=interactions['llm_id'].to_numpy(),
            llm_ids=llm_ids,
            top_k=config['top_response_docs'],
            clip_negative=config['clip_negative']
        )

    # HIERARCHICAL
    if 'H' in sources:
        llm_ids, centroids = load_or_create_centroids(
            interactions=interactions,
            embeddings=interaction_embeddings,
            cache_path=config['centroid_path'],
            n_centroids=config['n_centroids'],
            seed=config['seed'],
            n_init=config['kmeans_n_init']
        )

        source_scores['H'] = hierarchical_scores(
            query_embeddings=query_embeddings,
            interaction_embeddings=interaction_embeddings,
            interactions=interactions,
            llm_ids=llm_ids,
            centroids=centroids,
            n_local=config['n_local'],
            broad_weight=config['broad_weight'],
            local_weight=config['local_weight'],
            clip_negative=config['clip_negative']
        )
    return llm_ids, source_scores