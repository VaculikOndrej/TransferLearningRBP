import numpy as np
from numpy.random import seed
import sklearn.utils
import tensorflow 
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tokenizers

from .encoding import sec_struct_ohe


tensorflow.random.set_seed(73)
seed(73)
seed = 73


def load_and_prepare(train_data_path, tokenizer_path, length_long_seq, reduce_ratio=1):
    all_seqs_train = []
    all_labels_train = []
    all_cons_train = []
    all_secs_train = []

    with open(train_data_path, 'r') as inp:
        for line in inp.readlines():
            header, seq, cons, sec, label = line.strip().upper().split('\t')
            all_seqs_train.append(seq)
            all_labels_train.append(label)
            all_cons_train.append(cons)
            all_secs_train.append(sec[:-1])

    y_train_raw = np.array(all_labels_train).astype(int)
    corpus_train = [seq.upper() for seq in all_seqs_train]
    cons_train = np.array([ np.array([cons[1:-1].split(',')]).reshape(-1,1).astype(float) for cons in all_cons_train ])
    secs_train = sec_struct_ohe(all_secs_train)
    corpus_train, cons_train, secs_train, y_train_raw = sklearn.utils.shuffle(corpus_train, cons_train, secs_train, y_train_raw)

    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)

    embedded_sequences_train = [tokenizer.encode(seq).ids for seq in corpus_train]

    padded_sequences_train = pad_sequences(embedded_sequences_train, length_long_seq, padding='post', truncating='post')

    reduction = int(len(y_train_raw)*reduce_ratio)

    padded_sequences_train, cons_train, secs_train, y_train_raw = padded_sequences_train[:reduction], cons_train[:reduction], secs_train[:reduction], y_train_raw[:reduction]

    return padded_sequences_train, cons_train, secs_train, y_train_raw