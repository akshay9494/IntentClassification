from abc import ABC, abstractmethod
from core.IntentClassification.configuration import EmbeddingsConfigurations

class EmbeddingLoader(ABC):
    @abstractmethod
    def load_embedding(self):
        pass

    @abstractmethod
    def get_embedding(self, embeddings_model, word):
        pass
