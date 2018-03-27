from .LanguagePreprocessor import LanguagePreprocessor
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk.stem
from sklearn.feature_extraction.text import TfidfVectorizer
from utilities.StemmedTfidfVectorizer import StemmedTfidfVectorizer
import numpy as np


class FrenchLanguagePreprocessor(LanguagePreprocessor):
    def __init__(self, model_properties):
        super(FrenchLanguagePreprocessor, self).__init__(FrenchLanguagePreprocessor)


    def __prepare_tfidf(self, text):
        with open(self.config['FRENCH_STOPWORDS'], 'r', encoding='utf8') as f:
            french_stopwords = f.readlines()

        french_stopwords = [f.strip('\n') for f in french_stopwords]
        tfidf = StemmedTfidfVectorizer(ngram_range=(1, 3),
                                       stop_words=french_stopwords,
                                       decode_error='ignore').fit_transform(text)
        tfidf_input_data = tfidf.todense()
        tfidf_input_data = np.array(tfidf_input_data)
        return tfidf_input_data


    def preprocess_data(self, df):
        texts = df['Text'].tolist()
        labels = df['Label'].tolist()

        # label encode
        labels = self.__encode_labels(labels)
        # tokenize text
        nn_input = self.__tokenize_text(texts)
        # prepare tfidf
        tfidf_input = self.__prepare_tfidf(texts)

        # shuffle and split data
        return self.__train_val_split(nn_input=nn_input, tfidf_input=tfidf_input, labels=labels)


