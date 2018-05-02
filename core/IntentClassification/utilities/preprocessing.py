import re

class Preprocessing:
    def __init__(self):
        pass

    @staticmethod
    def expand_english_sentences_contractions(sentence):
        """
        expands sentences from i'll to i will, i've to i have etc...
        :param sentence: the sentence to expand
        :return: expanded sentence
        """
        sentence = re.sub('can\'t', 'can not', sentence)
        sentence = re.sub('\'ll', ' will', sentence)
        sentence = re.sub('\'s', ' ', sentence)
        sentence = re.sub('\'ve', ' have', sentence)
        sentence = re.sub('\'d', ' had', sentence)
        sentence = re.sub('n\'t', ' not', sentence)
        sentence = re.sub('\'m', ' am', sentence)
        sentence = re.sub('\'re', ' are', sentence)
        sentence = re.sub('\?', ' ?', sentence)
        sentence = re.sub('\,', ' ,', sentence)
        sentence = re.sub('\.', ' .', sentence)
        sentence = re.sub('\!', ' !', sentence)
        return sentence