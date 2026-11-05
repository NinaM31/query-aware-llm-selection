from sentence_transformers import SentenceTransformer
import numpy as np

class MiniLMEncoder:
    '''Encode text using a SentenceTransformer MiniLM model'''
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device='cuda', batch_size=256):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        # Load encoder once
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts):
        '''Encode a collection of texts into normalized embeddings'''
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)