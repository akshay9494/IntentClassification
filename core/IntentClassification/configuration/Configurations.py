import json
import os
from .DataSourceConfigurations import DataSourceConfigurations
from .ModelConfigurations import ModelConfigurations
from .EmbeddingsConfigurations import EmbeddingsConfigurations
from .LanguageConfigurations import LanguageConfigurations
from .TrainingOutputsConfigurations import TraininOutputsConfigurations

config_file = os.path.join(os.path.dirname(__file__), 'configurations.json')

class Configurations:
    def __init__(self):
        config = json.load(open(config_file))
        self.model_properties = ModelConfigurations(config['modelling'])
        self.embeddings_properties = EmbeddingsConfigurations(config['embeddings'])
        self.language_properties = LanguageConfigurations(config['language'])
        self.data_source_properties = DataSourceConfigurations(config['dataSource'])
        self.intent_home = config['intentHome']
        self.training_output_properties = TraininOutputsConfigurations(config['trainingOutputs'])


