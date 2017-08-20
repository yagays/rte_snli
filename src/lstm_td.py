#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM, TimeDistributed, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers import Merge
from keras.utils import np_utils
from keras import backend as K

from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint

from gensim.models import KeyedVectors

from src.util import load_data, load_embedding_index

np.random.seed(6162)

# =====arguments=====
parser = argparse.ArgumentParser()
parser.add_argument("--train_sampling", action="store_true", help="run sampled train data")
parser.add_argument("--embedding_dir", default="", help="path to GloVe embedding matrix")
parser.add_argument("--embedding_file", default="", help="GloVe file name")
parser.add_argument("--batch_size", type=int, default=512, help="batch size")
parser.add_argument("--nb_epochs", type=int, default=10, help="numbers of epochs to train for")
parser.add_argument("--lstm_dim", type=int, default=200, help="LSTM dim.")
parser.add_argument("--embedding_dim", type=int, default=200, help="embedding dim.")
parser.add_argument("--model_dir", default="model/", help="path to model")
opt = parser.parse_args()
print("Arguments: ", opt)

batch_size = opt.batch_size
nb_epochs = opt.nb_epochs
lstm_dim = opt.lstm_dim
embedding_dim = opt.embedding_dim

dt_str = datetime.now().strftime("%Y%m%d")
ut_str = datetime.now().strftime("%s")
exp_file_name = os.path.splitext(os.path.basename(__file__))[0]
exp_stamp = "{}.{}.{}_{}_{}_{}".format(ut_str,
                                       exp_file_name,
                                       batch_size,
                                       nb_epochs,
                                       lstm_dim,
                                       embedding_dim)

model_dir = os.path.join(opt.model_dir, dt_str)
if not os.path.isdir(model_dir):
    print(model_dir)
    os.mkdir(model_dir)
model_name = os.path.join(model_dir, exp_stamp + ".model.json")
model_weights_name = os.path.join(model_dir, exp_stamp + ".weight.h5")
model_metrics_name = os.path.join(model_dir, exp_stamp + ".metrics.json")

# =====data preprocess=====
X_train, y_train, X_dev, y_dev, X_test, y_test, tokenizer = load_data(train_sampling=opt.train_sampling)

# =====preapare embedding matrix=====
word_index = tokenizer.word_index
num_words = len(word_index)

embeddings_index = load_embedding_index(opt.embedding_dir, opt.embedding_file)

embedding_matrix = np.zeros((len(word_index) + 1, 200))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# =====LSTM model=====
model1 = Sequential()
model1.add(Embedding(num_words + 1,
                     embedding_dim,
                     weights=[embedding_matrix],
                     trainable=False))
model1.add(LSTM(lstm_dim, recurrent_dropout=0.5, dropout=0.5, return_sequences=True))
model1.add(TimeDistributed(Dense(100, activation="relu")))
model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(100,)))

model2 = Sequential()
model2.add(Embedding(num_words + 1,
                     embedding_dim,
                     weights=[embedding_matrix],
                     trainable=False))
model2.add(LSTM(lstm_dim, recurrent_dropout=0.5, dropout=0.5, return_sequences=True))
model2.add(TimeDistributed(Dense(100, activation="relu")))
model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(100,)))

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

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

print()
print("Test score:", score)
print("Test accuracy:", acc)

# =====save model and evaluation=====
model.save_weights(model_weights_name)
with open(model_name, "w") as f:
    f.write(model.to_json())

with open(model_metrics_name, "w") as f:
    json.dump(
        {
            "evaluation": {"test_score": score, "test_accuracy": acc},
            "parameter": vars(opt)
        },
        f)
