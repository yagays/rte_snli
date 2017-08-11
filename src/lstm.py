#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers import Merge
from keras.utils import np_utils

from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint

from gensim.models import KeyedVectors

np.random.seed(6162)

# =====data preprocess=====
train_df = pd.read_csv("data/train.tsv", sep="\t", header=0)
dev_df = pd.read_csv("data/dev.tsv", sep="\t", header=0)
test_df = pd.read_csv("data/test.tsv", sep="\t", header=0)

# rm y label "-" line and fillna
train_df = train_df[train_df["gold_label"] != "-"].fillna("")
dev_df = dev_df[dev_df["gold_label"] != "-"].fillna("")
test_df = test_df[test_df["gold_label"] != "-"].fillna("")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df["sentence1"])
tokenizer.fit_on_texts(train_df["sentence2"])
tokenizer.fit_on_texts(dev_df["sentence1"])
tokenizer.fit_on_texts(dev_df["sentence2"])
tokenizer.fit_on_texts(test_df["sentence1"])
tokenizer.fit_on_texts(test_df["sentence2"])

seq_train1 = tokenizer.texts_to_sequences(train_df["sentence1"])
seq_train2 = tokenizer.texts_to_sequences(train_df["sentence2"])
seq_dev1 = tokenizer.texts_to_sequences(dev_df["sentence1"])
seq_dev2 = tokenizer.texts_to_sequences(dev_df["sentence2"])
seq_test1 = tokenizer.texts_to_sequences(test_df["sentence1"])
seq_test2 = tokenizer.texts_to_sequences(test_df["sentence2"])

maxlen = 78

X_train1 = sequence.pad_sequences(seq_train1, maxlen=maxlen)
X_train2 = sequence.pad_sequences(seq_train2, maxlen=maxlen)
X_train = [X_train1, X_train2]

y_label = {"contradiction":0, "entailment":1, "neutral":2}
y_train = [y_label[i] for i in train_df["gold_label"]]
y_train = np_utils.to_categorical(y_train, 3)

X_dev1 = sequence.pad_sequences(seq_dev1, maxlen=maxlen)
X_dev2 = sequence.pad_sequences(seq_dev2, maxlen=maxlen)
X_dev = [X_dev1, X_dev2]

y_dev = [y_label[i] for i in dev_df["gold_label"]]
y_dev = np_utils.to_categorical(y_dev, 3)

X_test1 = sequence.pad_sequences(seq_test1, maxlen=maxlen)
X_test2 = sequence.pad_sequences(seq_test2, maxlen=maxlen)
X_test = [X_test1, X_test2]

y_test = [y_label[i] for i in test_df["gold_label"]]
y_test = np_utils.to_categorical(y_test, 3)


# =====preapare embedding matrix=====
word_index = tokenizer.word_index
num_words = len(word_index)

embeddings_index = {}
with open("~/dev/nlp/wordvectors_model/glove.6B/glove.6B.200d.txt") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs


embedding_matrix = np.zeros((len(word_index) + 1, 200))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# =====LSTM model=====
batch_size = 512
nb_epochs = 50
lstm_dim = 200
embedding_dim = 200
max_features = 1000

model1 = Sequential()
model1.add(Embedding(num_words + 1,
                     embedding_dim,
                     weights=[embedding_matrix],
                     trainable=False))
model1.add(LSTM(embedding_dim, recurrent_dropout=0.5, dropout=0.5))

model2 = Sequential()
model2.add(Embedding(num_words + 1,
                     embedding_dim,
                     weights=[embedding_matrix],
                     trainable=False))
model2.add(LSTM(embedding_dim, recurrent_dropout=0.5, dropout=0.5))

model = Sequential()
model.add(Merge([model1, model2], mode="concat"))
model.add(BatchNormalization())

model.add(Dense(300))
model.add(PReLU())
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(300))
model.add(PReLU())
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(300))
model.add(PReLU())
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(3, activation="sigmoid"))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"]
              )


model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=nb_epochs,
          validation_data=(X_dev, y_dev),
          shuffle=True,
          )

y_pred = model.predict_classes(X_dev, batch_size=batch_size).flatten()
score, acc = model.evaluate(X_dev, y_dev, batch_size=batch_size)

print()
print("Test score:", score)
print("Test accuracy:", acc)
