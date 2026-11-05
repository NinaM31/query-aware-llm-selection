from ir_measures import calc_aggregate, nDCG, RR, P

def evaluate_results(results, qrels):
    '''Evaluate retrieval results using standard IR metrics'''
    # Prepare run for ir_measures
    run = results.rename(columns={'llm_id': 'doc_id'})[['query_id', 'doc_id', 'rank']].copy()

    # Preserve the exact ranking produced by the retriever
    run['score'] = -run['rank'].astype(float)
    run = run[['query_id', 'doc_id', 'score'] ]

    # Prepare qrels
    qrels_ir = qrels.rename(columns={'llm_id': 'doc_id'})[['query_id', 'doc_id', 'relevance']]

    raw_metrics = calc_aggregate([nDCG @ 5, nDCG @ 10, P @ 5, P @ 10, RR], qrels_ir, run)
    metrics = {str(metric): value for metric, value in raw_metrics.items()}
    
    return metrics