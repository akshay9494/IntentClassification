from abc import ABC, abstractmethod
from configuration import EmbeddingsConfigurations

class EmbeddingLoader(ABC):
    @abstractmethod
    def load_embedding(self):
        pass

    @abstractmethod
    def get_embedding(self, embeddings_model, word):
        pass
