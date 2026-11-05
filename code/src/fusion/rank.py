import numpy as np
from scipy.stats import rankdata
from src.fusion.source_scores import compute_source_scores

def scores_to_ranks(scores):
    '''Convert score matrices to rank matrices Highest score receives rank 1'''
    return rankdata(-scores, method='average', axis=1).astype(np.float32)

def percentile_rows(scores):
    '''Convert each row into percentile ranks'''
    scores = np.asarray(scores, dtype=float)

    percentiles = rankdata(scores, method='average', axis=1) / scores.shape[1]
    return percentiles.astype(np.float32)

def equal_fusion(source_scores, sources):
    '''Equal aggregation using percentile ranks'''
    normalized_sources = []

    for source in sources:
        normalized = percentile_rows(source_scores[source])
        normalized_sources.append(normalized)

    fused_scores = np.mean(normalized_sources, axis=0)
    return fused_scores.astype(np.float32)
    
def rrf_fusion(source_scores, sources, k=60):
    '''
        Reciprocal Rank Fusion\\
        RRF(i) = sum 1 / (k + rank_s(i))
    '''
    fused_scores = np.zeros_like(source_scores[sources[0]], dtype=np.float32)

    for source in sources:
        ranks = scores_to_ranks(source_scores[source])
        fused_scores += 1.0 / (k + ranks)
        
    return fused_scores.astype(np.float32)


def isr_fusion(source_scores, sources):
    '''
        Inverse Square Rank Fusion\\
        ISR(i) = N(i) * sum 1 / rank_s(i)^2\\
        Since every P/R/H source currently ranks every LLM, N(i) equals the number of selected sources
    '''
    fused_scores = np.zeros_like(source_scores[sources[0]], dtype=np.float32)

    for source in sources:
        ranks = scores_to_ranks(source_scores[source])
        fused_scores += 1.0 / (ranks ** 2)
    fused_scores *= len(sources)
    return fused_scores.astype(np.float32)

def run_rank_fusion(interactions, interaction_embeddings, query_embeddings, config):
    '''Run Equal, RRF, or ISR over the selected sources'''
    sources = config['sources']
    llm_ids, source_scores = compute_source_scores(
        interactions=interactions,
        interaction_embeddings=interaction_embeddings,
        query_embeddings=query_embeddings,
        config=config,
        sources=sources
    )

    if config['representation'] == 'equal':
        fused_scores = equal_fusion(source_scores=source_scores, sources=sources)
    if config['representation'] == 'rrf':
        fused_scores = rrf_fusion(source_scores=source_scores, sources=sources, k=config.get('rrf_k', 60))
    elif config['representation'] == 'isr':
        fused_scores = isr_fusion(source_scores=source_scores, sources=sources)

    return llm_ids, fused_scores