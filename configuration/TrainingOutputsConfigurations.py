class TraininOutputsConfigurations:
    def __init__(self, training_outputs_config):
        self.tokenizer_name = training_outputs_config['tokenizerName']
        self.label_encoder_name = training_outputs_config['labelEncoderName']
