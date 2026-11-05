import pandas as pd

def profile_to_lexical_documents(profiles):
    '''Convert profile representations into lexical documents Each LLM profile becomes one searchable document'''
    documents = profiles.copy()

    # Combine all historical interactions of one LLM into one document
    documents['text'] = documents['interactions'].apply(lambda interactions: ' '.join(interactions))

    # Each profile corresponds to one Lucene document
    documents['doc_id'] = documents['llm_id']

    # The lexical index only needs doc_id, llm_id, and text
    documents = documents.drop(columns=['interactions'])

    return documents