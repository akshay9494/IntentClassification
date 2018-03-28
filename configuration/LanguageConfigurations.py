from .FrenchLanguageConfigurations import FrenchLanguageConfigurations
from .EnglishLanguageConfigurations import EnglishLanguageConfigurations

class LanguageConfigurations:
    def __init__(self, language_config):
        # self.language_properties = language_config
        if 'french' in language_config.keys():
            self.language = 'french'
            self.french_properties = FrenchLanguageConfigurations(language_config['french'])
        else:
            self.language = 'english'
            self.english_properties = EnglishLanguageConfigurations(language_config['english'])