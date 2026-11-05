import pandas as pd
import numpy as np

def build_profile_representation(df):
    '''
        Build one profile per LLM from its historical interactions Each interaction combines query + response for each LLM

        LLM A, [interaction1, interaction2, interaction3, ...]
        LLM B, [interaction1, interaction2, ...]
    '''
    data = df.copy()

    # Combine the query and response into one text per row
    data['interaction_text'] = 'Query: ' + data['query'] + '\nResponse: ' + data['response']

    # Combine all interaction_text per LLM 
    profiles = data.groupby('llm_id')['interaction_text'].apply(list).reset_index(name='interactions')
    return profiles

def build_mean_profile_vectors(interactions, embeddings):
    '''Build one dense profile vector per LLM by mean-pooling its historical interaction embeddings'''
    llm_ids = sorted(interactions['llm_id'].unique())
    embedding_dim = embeddings.shape[1]
    profile_vectors = np.zeros((len(llm_ids), embedding_dim), dtype=np.float32)
    
    for i, llm_id in enumerate(llm_ids):
        # Get interaction rows belonging to this LLM
        indices = np.flatnonzero(interactions['llm_id'].to_numpy() == llm_id)

        # Mean-pool all historical interaction embeddings
        profile = embeddings[indices].mean(axis=0)

        # L2-normalize so dot product = cosine similarity
        norm = np.linalg.norm(profile)
        if norm > 0: profile = profile / norm

        profile_vectors[i] = profile.astype(np.float32)  
    return llm_ids, profile_vectors