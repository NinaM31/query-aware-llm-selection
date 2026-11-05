from java.nio.file import Paths

from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause, TermQuery
from org.apache.lucene.store import FSDirectory


class LexicalRetriever:
    def __init__(self, index_path, analyzer, similarity):
        '''Open a Lucene index for lexical retrieval'''
        # Open the existing Lucene index
        self.directory = FSDirectory.open(Paths.get(str(index_path)))

        # Read the index
        self.reader = DirectoryReader.open(self.directory)

        # Create the searcher
        self.searcher = IndexSearcher(self.reader)

        # Set the scoring function
        self.searcher.setSimilarity(similarity)

        # Parse queries against the indexed text field
        self.parser = QueryParser('text', analyzer)

    def search(self, query, top_k):
        '''Search the index for one query'''
        # Convert the raw query into a Lucene query
        lucene_query = self.parser.parse(QueryParser.escape(query))

        # Retrieve the highest-scoring documents
        hits = self.searcher.search(lucene_query, top_k).scoreDocs

        stored_fields = self.searcher.storedFields()

        results = []
        for rank, hit in enumerate(hits, start=1):
            document = stored_fields.document(hit.doc)
            results.append({
                'rank': rank,
                'doc_id': document.get('doc_id'),
                'llm_id': document.get('llm_id'),
                'score': hit.score
            })
        return results

    def close(self):
        '''Close the Lucene index resources'''
        self.reader.close()
        self.directory.close()