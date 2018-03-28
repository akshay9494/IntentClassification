class EmbeddingsConfigurations:
    def __init__(self, embedding_config):
        self.embeddings_path = embedding_config['embeddingsPath']
        self.embeddings_dim = int(embedding_config['embeddingsDimension'])