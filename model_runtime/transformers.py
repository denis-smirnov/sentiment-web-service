import numpy as np
from sklearn.base import TransformerMixin
from nltk import TweetTokenizer
from collections import Counter


class TextToFreqRankTransformer(TransformerMixin):

    def __init__(self, dictionary_size: int=1000):
        if dictionary_size <= 0:
            raise ValueError("Dictionary size should be > 0")
        self.dictionary_size = dictionary_size
        self.token_rank = None

    def fit(self, X, y=None, **fit_params):
        tt = TweetTokenizer()
        token_frequency_counter = Counter()

        for text in X:
            token_frequency_counter.update(tt.tokenize(text))

        self.token_rank = {token_freq[0]: rank + 1  # Ranks start from 1
                           for rank, token_freq in enumerate(
            token_frequency_counter.most_common(self.dictionary_size))}

        return self

    # TODO: avoid double tokenization by overriding fit_transform
    def transform(self, X, **transform_params):
        tt = TweetTokenizer()
        x_transformed = []
        for row in X:
            x_transformed.append(np.array(
                [self.token_rank[token]
                 for token in tt.tokenize(row)
                 if token in self.token_rank.keys()]))
        return np.array(x_transformed)


class SequencePaddingTransformer(TransformerMixin):

    def __init__(self, sequence_length=100, pad_with=0):
        if sequence_length <= 0:
            raise ValueError("Sequence length should be > 0")
        self.pad_with = pad_with
        self.sequence_length = sequence_length

    def fit(self, X, y=None, **fit_params):
        # self.sequence_length = max([len(seq) for seq in X])
        return self

    def transform(self, X, **transform_params):
        x_transformed = []
        for row in X:
            if len(row) > 0:
                x_transformed.append(np.array(
                    row[:self.sequence_length] if len(row) >= self.sequence_length
                    else list(row) + list([self.pad_with]*(self.sequence_length-len(row)))))
            else:
                x_transformed.append(np.repeat(self.pad_with, self.sequence_length))
        return np.array(x_transformed)
