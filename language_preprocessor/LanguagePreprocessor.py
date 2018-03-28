from abc import ABC, abstractmethod
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

class LanguagePreprocessor(ABC):
    def __init__(self, model_properties):
        self.model_properties = model_properties

    def encode_labels(self, labels):
        self.num_classes = len(np.unique(labels))
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(labels)
        # you need to pickle labels to be able to use it while prediction
        labels = label_encoder.transform(labels)
        labels = to_categorical(labels)
        return labels


    def tokenize_text(self, text):
        tokenizer = Tokenizer(num_words=self.model_properties.max_num_of_words)
        # see if you need to add tokens from embeddings as well
        tokenizer.fit_on_texts(text)
        sequences = tokenizer.texts_to_sequences(text)
        self.word_index = tokenizer.word_index
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
    def preprocess_data(self, data):
        pass