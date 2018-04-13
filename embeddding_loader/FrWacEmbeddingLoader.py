from .EmbeddingLoader import EmbeddingLoader
from gensim.models.keyedvectors import KeyedVectors
import logging


class FrWacEmbeddingLoader(EmbeddingLoader):
    def __init__(self, embeddings_properties):
        super(FrWacEmbeddingLoader, self).__init__()
        self.embedding_properties = embeddings_properties

    def load_embedding(self):
        embeddings_model = KeyedVectors.load_word2vec_format(self.embedding_properties.embedding_path,
                                                             binary=True,
                                                             unicode_errors='ignore')
        self.vocab = list(embeddings_model.wv.vocab)
        logging.debug('Found {} many embedding vectors.'.format(len(self.vocab)))
        return embeddings_model

    def get_embedding(self, embeddings_model, word):
        """

        :param word: the word to get embeddings for
        :return: embedding vector if word exists else returns None
        """
        try:
            return embeddings_model.word_vec(word)
        except KeyError:
            return None
