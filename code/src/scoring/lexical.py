from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.search.similarities import BM25Similarity, ClassicSimilarity

def get_analyzer(name='standard'):
    '''Return a Lucene analyzer.'''
    if name == 'standard':
        return StandardAnalyzer()

def get_bm25(k1=0.9, b=0.4):
    '''Create a BM25 similarity function'''
    return BM25Similarity(k1, b)

def get_tf_idf():
    '''Create a tfidf similarity function'''
    return ClassicSimilarity()