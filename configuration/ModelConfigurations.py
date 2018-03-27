class ModelConfigurations:
    def __init__(self, model_config):
        self.max_num_of_words = model_config['maxNumberOfWords']
        self.max_length  = model_config['maxLength']
        self.validation_split = float(model_config['validationSplit'])