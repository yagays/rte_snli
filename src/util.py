#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pickle
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils


def load_data(train_sampling=False):
    # =====params=====
    maxlen = 78
    y_label = {"contradiction": 0, "neutral": 1, "entailment": 2}

    X_train_pickle_path = "data/X_train_tokenized.pkl"
    y_train_pickle_path = "data/y_train_tokenized.pkl"
    X_dev_pickle_path = "data/X_dev_tokenized.pkl"
    y_dev_pickle_path = "data/y_dev_tokenized.pkl"
    X_test_pickle_path = "data/X_test_tokenized.pkl"
    y_test_pickle_path = "data/y_test_tokenized.pkl"
    tokenizer_pickle_path = "data/tokenizer.pkl"
    pickle_list = [X_train_pickle_path,
                   y_train_pickle_path,
                   X_dev_pickle_path,
                   y_dev_pickle_path,
                   X_test_pickle_path,
                   y_test_pickle_path,
                   tokenizer_pickle_path]

    if all([os.path.exists(f) for f in pickle_list]):
        with open(X_train_pickle_path, "rb") as f:
            X_train = pickle.load(f)
        with open(y_train_pickle_path, "rb") as f:
            y_train = pickle.load(f)
        with open(X_dev_pickle_path, "rb") as f:
            X_dev = pickle.load(f)
        with open(y_dev_pickle_path, "rb") as f:
            y_dev = pickle.load(f)
        with open(X_test_pickle_path, "rb") as f:
            X_test = pickle.load(f)
        with open(y_test_pickle_path, "rb") as f:
            y_test = pickle.load(f)
        with open(tokenizer_pickle_path, "rb") as f:
            tokenizer = pickle.load(f)
    else:
        # =====data preprocess=====
        train_df = pd.read_csv("data/snli_1.0_train.txt", sep="\t", header=0)
        dev_df = pd.read_csv("data/snli_1.0_dev.txt", sep="\t", header=0)
        test_df = pd.read_csv("data/snli_1.0_test.txt", sep="\t", header=0)

        # rm "-" of y line and fillna
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

        X_train1 = sequence.pad_sequences(seq_train1, maxlen=maxlen)
        X_train2 = sequence.pad_sequences(seq_train2, maxlen=maxlen)
        X_train = [X_train1, X_train2]

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

        with open(X_train_pickle_path, "wb") as f:
            pickle.dump(X_train, f)
        with open(y_train_pickle_path, "wb") as f:
            pickle.dump(y_train, f)
        with open(X_dev_pickle_path, "wb") as f:
            pickle.dump(X_dev, f)
        with open(y_dev_pickle_path, "wb") as f:
            pickle.dump(y_dev, f)
        with open(X_test_pickle_path, "wb") as f:
            pickle.dump(X_test, f)
        with open(y_test_pickle_path, "wb") as f:
            pickle.dump(y_test, f)
        with open(tokenizer_pickle_path, "wb") as f:
            pickle.dump(tokenizer, f)

    if train_sampling:
        X_train[0] = X_train[0][:10000]
        X_train[1] = X_train[1][:10000]
        y_train = y_train[:10000]

    return X_train, y_train, X_dev, y_dev, X_test, y_test, tokenizer


def load_embedding_index(file_dir, file_name):
    file_path = os.path.join(file_dir, file_name)
    pickle_path = os.path.join(file_dir, os.path.splitext(file_name)[0] + ".pkl")

    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            embeddings_index = pickle.load(f)
    else:
        embeddings_index = {}
        with open(file_path) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                embeddings_index[word] = coefs
        with open(pickle_path, "wb") as f:
            pickle.dump(embeddings_index, f)
    return embeddings_index
