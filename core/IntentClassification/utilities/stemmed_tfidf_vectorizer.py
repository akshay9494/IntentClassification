from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

french_stemmer = nltk.stem.SnowballStemmer('french')
english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))