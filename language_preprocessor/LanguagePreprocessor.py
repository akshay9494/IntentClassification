from abc import ABC, abstractmethod
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import logging
import pickle
import os
import nltk


class LanguagePreprocessor(ABC):
    def __init__(self, model_properties):
        self.model_properties = model_properties

    def encode_labels(self, labels, pickle_path):
        self.num_classes = len(np.unique(labels))
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(labels)
        # you need to pickle labels to be able to use it while prediction
        pickle.dump(label_encoder, open(os.path.join(pickle_path, 'label_encoder.pkl'), 'wb'))
        labels = label_encoder.transform(labels)
        labels = to_categorical(labels)
        return labels


    def tokenize_text(self, text, pickle_path, embeddings_words=''):
        """
        pass the embeddings_words in the parameter if you want to create the embeddings matrix with these words
        :param text: list of training texts
        :param embeddings_words: string of embeddings keys
        :return:
        """
        tokenizer = Tokenizer(num_words=self.model_properties.max_num_of_words,
                              filters='')
        text = [' '.join(nltk.word_tokenize(t)) for t in text]
        # see if you need to add tokens from embeddings as well
        tokenizer.fit_on_texts(text + [embeddings_words])
        pickle.dump(tokenizer, open(os.path.join(pickle_path, 'tokenizer.pkl'), 'wb'))
        sequences = tokenizer.texts_to_sequences(text)
        self.word_index = tokenizer.word_index
        logging.info('Found {} unique tokens'.format(len(self.word_index)))
        nn_input = pad_sequences(sequences, maxlen=self.model_properties.max_length)
        return nn_input


    def train_val_split(self, nn_input, tfidf_input, labels):
        indices = np.arange(nn_input.shape[0])
        np.random.shuffle(indices)
        nn_input = nn_input[indices]
        tfidf_input = tfidf_input[indices]
        labels = labels[indices]

        num_validation_samples = int(self.model_properties.validation_split * nn_input.shape[0])

        nn_input_train = nn_input[:-num_validation_samples]
        tfidf_input_train = tfidf_input[:-num_validation_samples]
        labels_train = labels[:-num_validation_samples]

        nn_input_val = nn_input[-num_validation_samples:]
        tfidf_input_val = tfidf_input[-num_validation_samples:]
        labels_val = labels[-num_validation_samples:]

        return nn_input_train, tfidf_input_train, labels_train, nn_input_val, tfidf_input_val, labels_val


    @abstractmethod
    def preprocess_data(self, data, pickle_path, embeddings_vocab=None):
        pass