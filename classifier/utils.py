import re
from string import punctuation
import pickle

import numpy as np

import nltk
# nltk.download('punkt')

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class PolarityClassifier:

    def __init__(self, path_to_model, path_to_tokenizer):
        self.model = self.load_model(path_to_model)
        self.tokenizer = self.load_tokenizer(path_to_tokenizer)

    def text_preprocessor(self, tweet):
        # remove user mentions
        tweet = re.sub('\s*@[a-zA-Z0-9]*\s*', ' ', tweet)
        # remove signle character
        tweet = re.sub('\s+[a-zA-Z0-9]\s+', ' ', tweet)
        # remove hashtag sign
        tweet = re.sub('#', '', tweet)
        # remove underline
        tweet = re.sub('_', ' ', tweet)
        # remove dash
        tweet = re.sub('-', ' ', tweet)
        # translate &
        tweet = re.sub('&', ' and ', tweet)
        # lower
        tweet = tweet.lower()
        # remove punctuation
        tweet = ' '.join([token for token in nltk.word_tokenize(tweet) if token not in punctuation])
        return tweet

    def load_model(self, path_to_model):
        return keras.models.load_model(path_to_model)

    def load_tokenizer(self, path_to_tokenizer):
        with open(path_to_tokenizer, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer

    def transform_tweets(self, tweets):
        encoded_docs = self.tokenizer.texts_to_sequences(tweets)
        max_length = 50
        x_test = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
        return x_test

    def predict(self, tweets):
        tweets = list(map(self.text_preprocessor, tweets))
        x_test = self.transform_tweets(tweets)
        y_pred = self.model.predict(x_test)
        return np.round(y_pred)