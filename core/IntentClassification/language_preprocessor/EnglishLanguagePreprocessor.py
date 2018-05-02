from .LanguagePreprocessor import LanguagePreprocessor
from core.IntentClassification.utilities.stemmed_tfidf_vectorizer import StemmedTfidfVectorizer
import numpy as np
from core.IntentClassification.utilities.preprocessing import Preprocessing
import logging
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer



class EnglishLanguagePreprocessor(LanguagePreprocessor):
    def __init__(self, model_properties):
        super().__init__(model_properties)

    def __prepare_tfidf(self, text, pickle_path):
        tfidf_instance = CountVectorizer(stop_words='english')
        tfidf_instance.fit(text)
        pickle.dump(tfidf_instance, open(os.path.join(pickle_path, 'vectorizer.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
        tfidf = tfidf_instance.transform(text)
        tfidf_input_data = tfidf.todense()
        tfidf_input_data = np.array(tfidf_input_data)
        return tfidf_input_data


    def preprocess_data(self, df, pickle_path, embeddings_vocab=None):
        texts = df['Text'].tolist()
        logging.info('Expanding contractions.')
        texts = [Preprocessing.expand_english_sentences_contractions(s) for s in texts]
        labels = df['Label'].tolist()

        # label encode
        labels = super().encode_labels(labels, pickle_path=pickle_path)
        # tokenize text
        embeddings_words = ''
        if embeddings_vocab is not None:
            embeddings_words = ' '.join(embeddings_vocab)
        nn_input = super().tokenize_text(texts, pickle_path=pickle_path, embeddings_words=embeddings_words)
        # prepare tfidf
        tfidf_input = self.__prepare_tfidf(texts, pickle_path)
        # shuffle and split data
        return super().train_val_split(nn_input=nn_input, tfidf_input=tfidf_input, labels=labels)