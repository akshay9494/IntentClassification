from .LanguagePreprocessor import LanguagePreprocessor
from utilities.StemmedTfidfVectorizer import StemmedTfidfVectorizer
import numpy as np

class EnglishLanguagePreprocessor(LanguagePreprocessor):
    def __init__(self, model_properties):
        super(EnglishLanguagePreprocessor, self).__init__(model_properties)

    def __prepare_tfidf(self, text):
        tfidf = StemmedTfidfVectorizer(ngram_range=(1, 3),
                                       stop_words='english',
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
        nn_input_train, tfidf_input_train, labels_train, nn_input_val, tfidf_input_val, labels_val = self.\
            __train_val_split(nn_input=nn_input, tfidf_input=tfidf_input, labels=labels)