from core.IntentClassification.configuration import Configurations
import pickle
from keras.models import load_model
import os
import numpy as np
import logging
from keras.preprocessing.sequence import pad_sequences
import nltk
from core.IntentClassification.utilities.preprocessing import Preprocessing
from log4mongo.handlers import MongoHandler
from core.IntentClassification.configuration.Configurations import Configurations

config = Configurations()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(MongoHandler(host=config.log_properties.host, collection=config.log_properties.col_name))


class IntentClassifier:
    def __init__(self):
        self.config = Configurations()
        self.__initialize()

    def __find_directory(self, project_home):
        sub_directories = [os.path.join(project_home, d) for d in os.listdir(project_home)]
        latest_folder = max(sub_directories, key=os.path.getmtime)
        return latest_folder


    def __initialize(self):
        logger.info('Beginning Initialization Process.')
        directory_path = self.__find_directory(self.config.intent_home)
        self.model = load_model(os.path.join(directory_path, 'intent_classifier.h5'))
        self.label_encoder = pickle.load(open(os.path.join(directory_path, 'label_encoder.pkl'), 'rb'))
        self.vectorizer = pickle.load(open(os.path.join(directory_path, 'vectorizer.pkl'), 'rb'))
        self.tokenizer = pickle.load(open(os.path.join(directory_path, 'tokenizer.pkl'), 'rb'))

        # can be improved
        if type(self.model.input_shape) is tuple:
            self.model_trained_with_vectorizer = False
        else:
            self.model_trained_with_vectorizer = True
        logger.info('Model Initialized.')


    def get_intent(self, sentence):
        logger.info('Beginning Intent Classification for sentence: {}'.format(sentence))
        sentence = ' '.join(nltk.word_tokenize(sentence))
        sentence = Preprocessing.expand_english_sentences_contractions(sentence)
        sequences = self.tokenizer.texts_to_sequences([sentence])
        nn_input = pad_sequences(sequences, maxlen=self.config.model_properties.max_length)
        if self.model_trained_with_vectorizer:
            vectorized = self.vectorizer.transform([sentence])
            vectorized = vectorized.todense()
            vectorized = np.array(vectorized)
            prediction = self.label_encoder.inverse_transform(np.argmax(self.model.predict([nn_input, vectorized])))
        else:
            prediction = self.label_encoder.inverse_transform(np.argmax(self.model.predict([nn_input])))
        logger.info('Intent of {} is {}'.format(sentence, prediction))
        return prediction


if __name__ == '__main__':
    intent_classification_instance = IntentClassifier()
    while True:
        sentence = input('Enter your sentence: ')
        intent = intent_classification_instance.get_intent(sentence)
        print(intent)