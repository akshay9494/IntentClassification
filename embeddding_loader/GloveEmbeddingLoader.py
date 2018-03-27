from .EmbeddingLoader import EmbeddingLoader
from tqdm import tqdm
import numpy as np

class GloveEmbeddingLoader(EmbeddingLoader):
    def __init__(self, embeddings_properties):
        self.embedding_properties = embeddings_properties


    def load_embedding(self):
        """
        loads english embeddings (glove)
        :return: embedding dictionary consisiting of words and their vectors
        """
        embedding_index = {}
        f = open(self.embedding_properties.embedding_path, encoding='utf8')
        for line in tqdm(f):
            values = line.aplit()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
        f.close()
        return embedding_index

    def get_embedding(self, embeddings_model, word):
        """

        :param word: the word to get the embedding for
        :return: vector if exists else None
        """
        try:
            embedding_index.get(word)
        except KeyError:
            return None
