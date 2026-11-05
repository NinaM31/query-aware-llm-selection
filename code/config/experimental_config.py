import os
from copy import deepcopy
from pathlib import Path

SEED = 42
DATASET = os.environ.get('TREC_DATASET', '2026')
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_CONFIGS = {
    '2026': {
        'data_dir': PROJECT_ROOT / 'trec_mllm_2026',
        'cache_dir': PROJECT_ROOT / 'trec_mllm_2026' / 'cache',
        'results_dir': PROJECT_ROOT / 'results' / '2026',
        'n_candidates': 150,
    },
    '2025': {
        'data_dir': PROJECT_ROOT / 'trec_mllm_2025',
        'cache_dir': PROJECT_ROOT / 'trec_mllm_2025' / 'cache',
        'results_dir': PROJECT_ROOT / 'results' / '2025',
        'n_candidates': 1131,
    },
}

DATASET_CONFIG = DATASET_CONFIGS[DATASET]
DATA_DIR = DATASET_CONFIG['data_dir']
CACHE_DIR = DATASET_CONFIG['cache_dir']
RESULTS_DIR = DATASET_CONFIG['results_dir']
N_CANDIDATES = DATASET_CONFIG['n_candidates']

def make_config(base, **updates):
    config = deepcopy(base)
    config.update(updates)
    return config

# BoW retrieval: BM25, TF-IDF
BASE_LEXICAL_CONFIG = {
    'dataset': DATASET,
    'retrieval': 'lexical',
    'analyzer': 'standard',
    'top_k': N_CANDIDATES,
    'seed': SEED,
}

CONFIG_BM25 = make_config(
    BASE_LEXICAL_CONFIG,
    representation='profile',
    scoring='bm25',
    index_path=CACHE_DIR / 'lucene_profile_index',
    k1=0.9,
    b=0.4,
)

CONFIG_TFIDF = make_config(
    BASE_LEXICAL_CONFIG,
    representation='profile',
    scoring='tfidf',
    index_path=CACHE_DIR / 'lucene_profile_index',
)


# MiniLM retrieval: profile (P), response (R), hierarchical (H)
ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'
BASE_DENSE_CONFIG = {
    'dataset': DATASET,
    'encoder': ENCODER,
    'device': 'cuda',
    'batch_size': 256,
    'seed': SEED,
    'interaction_embedding_path': CACHE_DIR / 'minilm_interactions.npy',
    'query_embedding_path': CACHE_DIR / 'minilm_dev_queries.npy',
    'retrieval': 'dense',
    'scoring': 'cosine',
    'clip_negative': True,
    'top_k': N_CANDIDATES,
    'top_response_docs': 1000,
    'centroid_path': CACHE_DIR / 'minilm_kmeans10_seed42.npy',
    'n_centroids': 10,
    'kmeans_n_init': 10,
    'n_local': 10,
    'broad_weight': 0.8,
    'local_weight': 0.2,
}

CONFIG_P = make_config(BASE_DENSE_CONFIG, representation='profile')
CONFIG_R = make_config(BASE_DENSE_CONFIG, representation='response')
CONFIG_H = make_config(BASE_DENSE_CONFIG, representation='hierarchical')


# Equal: three pairwise combinations plus P + R + H
FUSION_SOURCE_COMBINATIONS = {
    'RH': ['R', 'H'],
    'RP': ['R', 'P'],
    'HP': ['H', 'P'],
    'RHP': ['R', 'H', 'P'],
}

EQUAL_CONFIGS = {
    name: make_config(
        BASE_DENSE_CONFIG,
        representation='equal',
        sources=sources,
        normalization='percentile',
        temperature=1.0,
    )
    for name, sources in FUSION_SOURCE_COMBINATIONS.items()
}

# ISR and RRF: only the three-source RHP combination
CONFIG_ISR_RHP = make_config(
    BASE_DENSE_CONFIG,
    representation='isr',
    sources=['R', 'H', 'P'],
)

CONFIG_RRF_RHP = make_config(
    BASE_DENSE_CONFIG,
    representation='rrf',
    sources=['R', 'H', 'P'],
    rrf_k=60,
)

EXPERIMENTS = {
    'BM25': CONFIG_BM25,
    'TF-IDF': CONFIG_TFIDF,
    'P': CONFIG_P,
    'R': CONFIG_R,
    'H': CONFIG_H,
    **{f'Equal-{name}': config for name, config in EQUAL_CONFIGS.items()},
    'ISR-RHP': CONFIG_ISR_RHP,
    'RRF-RHP': CONFIG_RRF_RHP,
}