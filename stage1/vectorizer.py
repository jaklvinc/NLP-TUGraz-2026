import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import gensim.downloader as api

class FastTextVectorizer(BaseEstimator,TransformerMixin):
    _global_model = None
    _global_cache = {}

    def __init__(self,model_name="fasttext-wiki-news-subwords-300"):
        self.model_name = model_name

    def fit(self, X, y=None):
        if FastTextVectorizer._global_model is None:
            print(f"Loading {self.model_name}")
            FastTextVectorizer._global_model = api.load(self.model_name)
        return self

    def transform(self, X, y=None):
        model = FastTextVectorizer._global_model
        vectors = []

        for text in X:
            if text in FastTextVectorizer._global_cache:
                vectors.append(FastTextVectorizer._global_cache[text])
                continue

            clean_text = re.sub(r'[\W_]+',' ',str(text).lower())
            words = clean_text.split()

            word_vecs = []
            for w in words:
                try:
                    word_vecs.append(model[w])
                except KeyError:
                    pass
            sentence_vec = np.mean(word_vecs, axis=0)
            FastTextVectorizer._global_cache[text] = sentence_vec
            vectors.append(sentence_vec)

        return np.vstack(vectors)