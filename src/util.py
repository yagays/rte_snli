#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils


def load_data(train_sampling=False):
    # =====params=====
    maxlen = 78
    y_label = {"contradiction": 0, "neutral": 1, "entailment": 2}

    # =====data preprocess=====
    train_df = pd.read_csv("data/train.tsv", sep="\t", header=0)
    dev_df = pd.read_csv("data/dev.tsv", sep="\t", header=0)
    test_df = pd.read_csv("data/test.tsv", sep="\t", header=0)

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

    # sampling train data after tokenize all words in train_df
    if train_sampling:
        train_df = train_df.sample(10000, random_state=6162)

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

    return X_train, y_train, X_dev, y_dev, X_test, y_test, tokenizer


def load_embedding_index(file_dir, file_name):
    embeddings_index = {}
    with open(os.path.join(file_dir, file_name)) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    return embeddings_index
