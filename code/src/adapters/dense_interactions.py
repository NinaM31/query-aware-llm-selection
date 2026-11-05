import pandas as pd

def historical_to_dense_interactions(historical_data):
    '''Convert historical query-response pairs into text units for dense encoding'''
    data = historical_data.copy()

    # Each historical query-response pair is one interaction
    data['doc_id'] = [f'resp_{i}' for i in range(len(data))]
    data['text'] = 'Query: ' + data['query'] + '\nResponse: ' + data['response']
    
    return data[['doc_id', 'llm_id', 'text']]