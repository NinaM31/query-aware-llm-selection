import numpy as np
from src.representations.hierarchical import build_kmeans_centroids

def load_or_create_centroids(interactions, embeddings, cache_path, n_centroids=10, seed=42,n_init=10):
    '''Load cached centroids if they exist, otherwise build and save them'''
    llm_ids = sorted(interactions['llm_id'].unique())

    if cache_path.exists():
        centroids = np.load(cache_path)

        print(f'Loaded centroids: {cache_path.name} {centroids.shape}')
        return llm_ids, centroids

    llm_ids, centroids = build_kmeans_centroids(
        interactions=interactions,
        embeddings=embeddings,
        n_centroids=n_centroids,
        seed=seed,
        n_init=n_init
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, centroids)
    
    print(f'Saved centroids: {cache_path.name} {centroids.shape}')
    
    return llm_ids, centroids
    
def load_or_create_embeddings(texts, encoder, cache_path):
    '''Load embeddings from disk if they exist, otherwise compute and save them'''
    if cache_path.exists():
        embeddings = np.load(cache_path)
        print(f'Loaded embeddings: {cache_path.name} {embeddings.shape}')
        return embeddings
        
    embeddings = encoder.encode(texts)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)

    print(f'Saved embeddings: {cache_path.name} {embeddings.shape}')
    return embeddings