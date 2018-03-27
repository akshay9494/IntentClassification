import numpy as np
from configuration import Configurations
import embeddding_loader
import data_loader
import language_preprocessor


class IntentTrainer():
    def __init__(self, embedding_loader, data_loader, language_preprocessor, configurations):
        self.embedding_loader_instance = embedding_loader
        self.data_loader_instance = data_loader
        self.language_preprocessor_instance = language_preprocessor
        self.config = configurations


    def __create_embedding_matrix(self, embeddings):
        num_words = min(len(self.language_preprocessor_instance.word_index)+1, self.config.model_properties.max_num_of_words)
        embedding_matrix = np.zeros((num_words, self.config.embeddings_properties.embeddings_dim))
        words_with_no_embeddings = []
        for word, i in self.language_preprocessor_instance.word_index.items():
            if i >= self.config.model_properties.max_num_of_words:
                continue
            embedding_vector = self.embedding_loader_instance.get_embedding(embeddings_model=embeddings,
                                                                            word=word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                words_with_no_embeddings.append(word)
        return embedding_matrix


    def __load_embeddings(self):
        return self.embedding_loader_instance.load_embedding()


    def __load_data(self):
        data = self.data_loader_instance.load_data()
        return data


    def __preprocess_data(self, data):
        return self.language_preprocessor_instance.preprocess_data(data)


    def train(self):
        data = self.__load_data()
        embeddings = self.__load_embeddings()
        nn_input_train, tfidf_input_train, labels_train, nn_input_val, tfidf_input_val, labels_val = self.\
            __preprocess_data(data)
        embedding_matrix = self.__create_embedding_matrix(embeddings)
        # now model can be trained


if __name__ == '__main__':
    config = Configurations()
    if config.language_properties.language == 'french':
        embeddding_loader_instance = embeddding_loader.FrWacEmbeddingLoader(embeddings_properties=
                                                                            config.embeddings_properties)
        language_preprocessor_instance = language_preprocessor.FrenchLanguagePreprocessor(model_properties=
                                                                                          config.model_properties)
    else:
        embeddding_loader_instance = embeddding_loader.GloveEmbeddingLoader(embeddings_properties=
                                                                            config.embeddings_properties)
        language_preprocessor_instance = language_preprocessor.EnglishLanguagePreprocessor(model_properties=
                                                                                           config.model_properties)

    if config.data_source_properties.data_source == 'db':
        data_loader_instance = data_loader.DBDataLoader(db_properties=
                                                        config.data_source_properties.db_configurations)
    else:
        data_loader_instance = data_loader.FileDataLoader(file_properties=
                                                          config.data_source_properties.file_configurations)
