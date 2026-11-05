import pandas as pd
import numpy as np
import random
import torch
import lucene

# Initialize the JVM once
if lucene.getVMEnv() is None:
    lucene.initVM()

# Data representations
from src.representations.profile import build_profile_representation, build_mean_profile_vectors
from src.representations.response import response_combsum_scores
from src.representations.hierarchical import hierarchical_scores

# Lexical retrieval
from src.adapters.lexical_documents import profile_to_lexical_documents
from src.scoring.lexical import get_analyzer, get_bm25, get_tf_idf
from src.indexes.lucene_text import build_lucene_inverted_index
from src.retrievers.lexical import LexicalRetriever

# Dense retrieval
from src.adapters.dense_interactions import historical_to_dense_interactions
from src.encoders.minilm import MiniLMEncoder
from src.retrievers.dense import DenseRetriever

# Cache
from src.cache.cache import load_or_create_embeddings

# Rank aggregation
from src.fusion.rank import run_rank_fusion


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def run_experiment(historical_data, test_queries, config):
    '''Run one retrieval experiment :)'''
    set_seed(config['seed'])

    # LEXICAL RETRIEVAL: PROFILE BM25 / TF-IDF
    if config['retrieval'] == 'lexical':
        profiles = build_profile_representation(historical_data)
        documents = profile_to_lexical_documents(profiles)

        # Select analyzer
        analyzer = get_analyzer(config['analyzer'])
        
        # Build Lucene index
        build_lucene_inverted_index(documents=documents, analyzer=analyzer, index_path=config['index_path'])

        # Select scoring function
        if config['scoring'] == 'bm25': similarity = get_bm25(k1=config['k1'], b=config['b'])
        if config['scoring'] == 'tfidf': similarity = get_tf_idf()

        # Open retriever 
        retriever = LexicalRetriever(index_path=config['index_path'], analyzer=analyzer, similarity=similarity)

        results = []
        try:
            for _, row in test_queries.iterrows():
                ranking = retriever.search(query=row['query'], top_k=config['top_k'])
                
                for result in ranking:
                    results.append({
                        'query_id': row['query_id'],
                        'llm_id': result['llm_id'],
                        'rank': result['rank'],
                        'score': result['score']
                    })
        finally:
            retriever.close()
        return pd.DataFrame(results)

    # DENSE RETRIEVAL
    if config['retrieval'] == 'dense':
        # Prepare historical interaction texts
        interactions = historical_to_dense_interactions(historical_data)

        # Load encoder
        encoder = MiniLMEncoder(model_name=config['encoder'], device=config['device'], batch_size=config['batch_size'])

        # Load or create shared historical embeddings
        interaction_embeddings = load_or_create_embeddings(
            texts=interactions['text'].tolist(), 
            encoder=encoder, 
            cache_path=config['interaction_embedding_path']
        )

        # Load or create shared query embeddings
        query_embeddings = load_or_create_embeddings(
            texts=test_queries['query'].tolist(),
            encoder=encoder,
            cache_path=config['query_embedding_path']
        )

        llm_ids = sorted(historical_data['llm_id'].unique())

        # PROFILE
        if config['representation'] == 'profile':
            llm_ids, profile_vectors = build_mean_profile_vectors(interactions, interaction_embeddings)
            retriever = DenseRetriever(candidate_ids=llm_ids, candidate_embeddings=profile_vectors)

            results = []
            for query_idx, (_, row) in enumerate(test_queries.iterrows()):
                ranking = retriever.search(
                    query_embedding=query_embeddings[query_idx],
                    top_k=config['top_k']
                )

                for result in ranking:
                    results.append({
                        'query_id': row['query_id'],
                        'llm_id': result['llm_id'],
                        'rank': result['rank'],
                        'score': result['score']
                    })
            return pd.DataFrame(results)

        # RESPONSE
        if config['representation'] == 'response':
            response_scores = response_combsum_scores(
                query_embeddings=query_embeddings,
                interaction_embeddings=interaction_embeddings,
                interaction_llm_ids=interactions['llm_id'].to_numpy(),
                llm_ids=llm_ids,
                top_k=config['top_response_docs'],
                clip_negative=config['clip_negative']
            )

            results = []
            for query_idx, (_, row) in enumerate(test_queries.iterrows()):
                scores = response_scores[query_idx]
                order = np.argsort(-scores, kind='stable')

                for rank, llm_idx in enumerate(order, start=1):
                    results.append({
                        'query_id': row['query_id'],
                        'llm_id': llm_ids[llm_idx],
                        'rank': rank,
                        'score': float(scores[llm_idx])
                    })
            return pd.DataFrame(results)

        # HIERARCHICAL
        if config['representation'] == 'hierarchical':
            # Load or build KMeans centroids
            llm_ids, centroids = load_or_create_centroids(
                interactions=interactions,
                embeddings=interaction_embeddings,
                cache_path=config['centroid_path'],
                n_centroids=config['n_centroids'],
                seed=config['seed'],
                n_init=config['kmeans_n_init']
            )

            # Compute H scores
            H_scores = hierarchical_scores(
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

            results = []
            for query_idx, (_, row) in enumerate(test_queries.iterrows()):
                scores = H_scores[query_idx]
                order = np.argsort(-scores, kind='stable')

                for rank, llm_idx in enumerate(order, start=1):
                    results.append({
                        'query_id': row['query_id'],
                        'llm_id': llm_ids[llm_idx],
                        'rank': rank,
                        'score': float(scores[llm_idx])
                    })
            return pd.DataFrame(results)

        # RANK AGGREGATION: EQUAL / RRF / ISR
        if config['representation'] in ['equal', 'rrf', 'isr']:
            llm_ids, fused_scores = run_rank_fusion(
                interactions=interactions,
                interaction_embeddings=interaction_embeddings,
                query_embeddings=query_embeddings,
                config=config
            )

            results = []
            for query_idx, (_, row) in enumerate(test_queries.iterrows()):
                scores = fused_scores[query_idx]
                order = np.argsort(-scores, kind='stable')

                for rank, llm_idx in enumerate(order, start=1):
                    results.append({
                        'query_id': row['query_id'],
                        'llm_id': llm_ids[llm_idx],
                        'rank': rank,
                        'score': float(scores[llm_idx])
                    })

            return pd.DataFrame(results)