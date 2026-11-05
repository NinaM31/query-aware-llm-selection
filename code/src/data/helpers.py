import pandas as pd

def load_trec_data(path):
    '''Load and unify a TREC Million LLM dataset'''
    data_path = path / 'data'
    
    # Historical discovery interactions
    historical_data = pd.read_parquet(data_path / 'llm_discovery_data_filtered.parquet')

    # Development queries
    test_queries = pd.read_csv(
        data_path / 'llm_dev_data.tsv',
        sep='\t',
        header=None,
        names=['query_id', 'query']
    )

    # Relevance judgments
    qrels = pd.read_csv(
        data_path / 'llm_dev_qrels.txt',
        sep=r'\s+',
        header=None,
        names=['query_id', 'iteration', 'llm_id', 'relevance']
    )

    # Remove TREC-specific column we do not need
    qrels = qrels.drop(columns=['iteration'])

    # Standardize historical data
    historical_data['query_id'] = historical_data['query_id'].astype(str)
    historical_data['llm_id'] = historical_data['llm_id'].astype(str)
    historical_data['query'] = historical_data['query'].astype(str)
    historical_data['response'] = historical_data['response'].fillna('').astype(str)

    # Standardize test queries
    test_queries['query_id'] = test_queries['query_id'].astype(str)
    test_queries['query'] = test_queries['query'].astype(str)

    # Standardize qrels
    qrels['query_id'] = qrels['query_id'].astype(str)
    qrels['llm_id'] = qrels['llm_id'].astype(str)
    qrels['relevance'] = qrels['relevance'].astype(int)

    return historical_data, test_queries, qrels