class ModelConfigurations:
    def __init__(self, model_config):
        self.max_num_of_words = model_config['maxNumberOfWords']
        self.max_length = model_config['maxLength']
        self.validation_split = model_config['validationSplit']
        self.batch_size = model_config['batchSize']
        self.num_iterations = model_config['numIterations']
        self.dropout = model_config['dropout']
        self.learning_rate = model_config['learningRate']
        self.model_identifier = model_config['modelIdentifier']
        self.use_embedding_vocab = model_config['useEmbeddingVocab']