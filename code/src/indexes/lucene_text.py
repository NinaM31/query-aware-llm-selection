from java.nio.file import Paths

from org.apache.lucene.document import Document, Field, StringField, TextField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import FSDirectory


def build_lucene_inverted_index(documents, analyzer, index_path, similarity=None):
    '''Build a Lucene inverted index from text documents'''
    # Open/create the directory where the Lucene index will be stored
    directory = FSDirectory.open(Paths.get(str(index_path)))

    # Configure how Lucene should analyze text while building the index
    config = IndexWriterConfig(analyzer)
    if similarity is not None:
        config.setSimilarity(similarity)

    # Rebuild the index from scratch
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)

    # Create the object that writes documents into the index
    writer = IndexWriter(directory, config)

    # Add each document to the Lucene index
    for _, row in documents.iterrows():

        document = Document()

        # Unique identifier for this indexed document
        document.add(StringField('doc_id', str(row['doc_id']), Field.Store.YES))

        # LLM that this document belongs to
        document.add(StringField('llm_id', str(row['llm_id']), Field.Store.YES))

        # Text that will be analyzed and indexed for retrieval
        document.add(TextField('text', row['text'], Field.Store.NO))

        writer.addDocument(document)

    writer.close()
    directory.close()