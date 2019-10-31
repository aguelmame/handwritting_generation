from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, allow_multiple=True):
        self.allow_multiple = allow_multiple

    def fit(self, X, y=None):
        """
        X is a list of strings
        """
        vocab = set("".join(X))
        self.vocab = dict(zip(vocab, range(len(vocab))))
        return self

    def transform(self, X):
        """
        X is a list of strings
        """
        encoded = np.zeros((len(X), len(self.vocab)))
        for i, text in enumerate(X):
            if self.allow_multiple:
                voc = Counter(text)
                for letter, n in voc.items():
                    encoded[i][self.vocab[letter]] = n
            else:
                voc = set(text)
                for letter in voc:
                    encoded[i][self.vocab[letter]] = 1

        return encoded


if __name__ == '__main__':
    texts = [
        'welcome',
        'elwcome',
        'elias',
        'test',
    ]

    enc = OneHotEncoder()
    out = enc.fit_transform(texts)
    print(out)
