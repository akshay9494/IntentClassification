import numpy as np
from configuration import Configurations
import embeddding_loader
import data_loader
import language_preprocessor
from utilities.model_architectures import ModelArchitectures
from keras import backend as K
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class IntentTrainer():
    def __init__(self, embedding_loader, data_loader, language_preprocessor, configurations):
        self.embedding_loader_instance = embedding_loader
        self.data_loader_instance = data_loader
        self.language_preprocessor_instance = language_preprocessor
        self.config = configurations


    def __create_embedding_matrix(self, embeddings):
        num_words = min(len(self.language_preprocessor_instance.word_index)+1, self.config.model_properties.max_num_of_words)
        logging.debug('Number of words for embeddings: {}'.format(num_words))
        embedding_matrix = np.zeros((num_words, self.config.embeddings_properties.embeddings_dim))
        logging.debug('Shape of Embedding Matrix: {}'.format(embedding_matrix.shape))
        words_with_no_embeddings = []
        for word, i in self.language_preprocessor_instance.word_index.items():
            if i >= self.config.model_properties.max_num_of_words:
                continue
            embedding_vector = self.embedding_loader_instance.get_embedding(embeddings_model=embeddings,
                                                                            word=word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                logging.debug('No Embedding Found for: {}'.format(word))
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
        logging.info('Loading Data For the model.')
        data = self.__load_data()

        logging.info('Preprocessing Data For the model.')
        nn_input_train, tfidf_input_train, labels_train, nn_input_val, tfidf_input_val, labels_val = self.\
            __preprocess_data(data)

        logging.info('Loading Embeddings.')
        embeddings = self.__load_embeddings()

        logging.info('Creating Embeddings Matrix.')
        embedding_matrix = self.__create_embedding_matrix(embeddings)

        logging.info('Loading Model Architecture.')
        # now model can be trained
        K.clear_session()
        model_architecture = ModelArchitectures(model_properties=
                                                self.config.model_properties).get_model()

        # add callbacks for model checkpointing, tensorboard etc...
        callbacks = []

        if 'tfidf' in self.config.model_properties.model_identifier:
            model = model_architecture(embedding_matrix=embedding_matrix,
                                       word_index=self.language_preprocessor_instance.word_index,
                                       input_shape=tfidf_input_train.shape[1],
                                       num_classes=self.language_preprocessor_instance.num_classes)
            print(model.summary())
            model.fit([nn_input_train, tfidf_input_train], labels_train,
                      validation_data=([nn_input_val, tfidf_input_val], labels_val),
                      epochs=self.config.model_properties.num_iterations,
                      batch_size=self.config.model_properties.batch_size,
                      callbacks=callbacks)
        else:
            model = model_architecture(embedding_matrix=embedding_matrix,
                                       word_index=self.language_preprocessor_instance.word_index,
                                       num_classes=self.language_preprocessor_instance.num_classes)
            print(model.summary())
            model.fit(nn_input_train, labels_train,
                      validation_data=(nn_input_val, labels_val),
                      epochs=self.config.model_properties.num_iterations,
                      batch_size=self.config.model_properties.batch_size,
                      callbacks=callbacks)

        logging.info('Training Complete.')




if __name__ == '__main__':
    logging.info('Loading Configurations')
    config = Configurations()
    logging.info('Language properties being set for {}'.format(config.language_properties.language))
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

    logging.info('Data Source properties being set for {}'.format(config.data_source_properties.data_source))
    if config.data_source_properties.data_source == 'db':
        data_loader_instance = data_loader.DBDataLoader(db_properties=
                                                        config.data_source_properties.db_configurations)
    else:
        data_loader_instance = data_loader.FileDataLoader(file_properties=
                                                          config.data_source_properties.file_properties)

    logging.info('Creating Instance for Intent Trainer.')
    intent_trainer = IntentTrainer(embedding_loader=embeddding_loader_instance,
                                   data_loader=data_loader_instance,
                                   language_preprocessor=language_preprocessor_instance,
                                   configurations=config)
    logging.info('Beginning Training.')
    intent_trainer.train()