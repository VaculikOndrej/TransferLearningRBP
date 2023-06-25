from IPython.core.display import HTML
import numpy as np
from numpy.random import seed
import os
import pandas as pd
from pathlib import Path
import random
import seaborn as sns
import tensorflow 
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (Input, Conv1D, Dense, Flatten, BatchNormalization, Dropout, Embedding, SpatialDropout1D, GlobalAveragePooling1D, Bidirectional, GRU, Activation, concatenate)
import tokenizers
from tokenizers import Tokenizer, normalizers, models, pre_tokenizers, decoders, trainers, processors
from statistics import mean

from .att_layer import Attention


def build_params_complete_model(cons_tr, 
                        secs_tr, 
                        n_feature_maps=32, 
                        gru_units=64, 
                        dense_lays=2,
                        emb=16,
                        resnet_blocks=2,
                        words=16,
                        length_long_seq=150):

    # Define input tensor
    # inp = Input(shape=(X_train.shape[1],), dtype='int32')
    int_sequences_input = Input(shape=(length_long_seq,))
    # Word embedding layer
    # embedded_inputs = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable = True)(inp)
    embedded_inputs = Embedding(input_dim=words, output_dim=emb, input_length=length_long_seq)(int_sequences_input)

    # Apply dropout to prevent overfitting
    embedded_inputs = SpatialDropout1D(0.25)(embedded_inputs)

    # Apply Conv1D
    conv1d = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(embedded_inputs)
    conv1d = Dropout(0.25)(conv1d)
    conv1d = BatchNormalization()(conv1d)

    # Apply Bidirectional GRU over embedded inputs
    rnn_outs = Bidirectional(GRU(gru_units, return_sequences=True))(conv1d)
    rnn_outs = Dropout(0.25)(rnn_outs) # Apply dropout to GRU outputs to prevent overfitting

    # Attention Mechanism - Generate attention vectors
    seq, word_scores = Attention(return_attention=True, name = "att")(rnn_outs)

    input_shape_cons = (cons_tr.shape[1], cons_tr.shape[2])
    cons_sequences_input = Input(input_shape_cons)

    # BLOCK 1

    conv_x = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(cons_sequences_input)
    conv_x = Dropout(.25)(conv_x)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = Dropout(.25)(conv_y)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = Dropout(.25)(conv_z)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(cons_sequences_input)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_1 = tensorflow.keras.layers.add([shortcut_y, conv_z])
    output_block_1 = Activation('relu')(output_block_1)

    if resnet_blocks == 2:

        # BLOCK 2

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_final = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_final = Activation('relu')(output_block_final)

    elif resnet_blocks == 3:
  
        # BLOCK 2

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_2 = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_2 = Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = BatchNormalization()(output_block_2)

        output_block_final = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_final = Activation('relu')(output_block_final)

    elif resnet_blocks == 4:
  
        # BLOCK 2

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_2 = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_2 = Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_2)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_3 = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_3 = Activation('relu')(output_block_3)

        # BLOCK 4

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_3)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = BatchNormalization()(output_block_3)

        output_block_final = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_final = Activation('relu')(output_block_final)
    
    else:
        print('Number of resnet blocks must be between 2 and 4')

    # FINAL
    gap_layer = GlobalAveragePooling1D()(output_block_final)

    # model.add(Flatten())
    flatten_1 = Flatten()(gap_layer)


    input_shape_sec = (secs_tr.shape[1], secs_tr.shape[2])
    secs_sequences_input = Input(input_shape_sec)

    # BLOCK 1

    conv_i = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(secs_sequences_input)
    conv_i = Dropout(.25)(conv_i)
    conv_i = BatchNormalization()(conv_i)
    conv_i = Activation('relu')(conv_i)

    conv_j = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_i)
    conv_j = Dropout(.25)(conv_j)
    conv_j = BatchNormalization()(conv_j)
    conv_j = Activation('relu')(conv_j)

    conv_k = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_j)
    conv_k = Dropout(.25)(conv_k)
    conv_k = BatchNormalization()(conv_k)

    # expand channels for the sum
    shortcut_u = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(secs_sequences_input)
    shortcut_u = BatchNormalization()(shortcut_u)

    output_block_1_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
    output_block_1_s = Activation('relu')(output_block_1_s)

    if resnet_blocks == 2:

        # BLOCK 2

        conv_i = Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # expand channels for the sum
        shortcut_u = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1_s)
        shortcut_u = BatchNormalization()(shortcut_u)

        output_block_final_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_final_s = Activation('relu')(output_block_final_s)

    elif resnet_blocks == 3:
        # BLOCK 2

        conv_i = Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # expand channels for the sum
        shortcut_u = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1_s)
        shortcut_u = BatchNormalization()(shortcut_u)

        output_block_2_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_2_s = Activation('relu')(output_block_2_s)

        # BLOCK 3

        conv_i = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # no need to expand channels because they are equal
        shortcut_u = BatchNormalization()(output_block_2_s)

        output_block_final_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_final_s = Activation('relu')(output_block_final_s)

    elif resnet_blocks == 4:
        # BLOCK 2

        conv_i = Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # expand channels for the sum
        shortcut_u = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1_s)
        shortcut_u = BatchNormalization()(shortcut_u)

        output_block_2_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_2_s = Activation('relu')(output_block_2_s)

        # BLOCK 3

        conv_i = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # no need to expand channels because they are equal
        shortcut_u = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_2_s)
        shortcut_u = BatchNormalization()(shortcut_u)

        output_block_3_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_3_s = Activation('relu')(output_block_3_s)

        # BLOCK 4

        conv_i = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_3_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # no need to expand channels because they are equal
        shortcut_u = BatchNormalization()(output_block_3_s)

        output_block_final_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_final_s = Activation('relu')(output_block_final_s)

    # FINAL
    gap_layer_s = GlobalAveragePooling1D()(output_block_final_s)

    # model.add(Flatten())
    flatten_2 = Flatten()(gap_layer_s)

    concat = concatenate([seq, flatten_1, flatten_2])

    if dense_lays == 2:
        # Dense layers
        fc1 = Dense(128, activation='relu')(concat)
        fc1 = Dropout(0.25)(fc1)
        fc1 = BatchNormalization()(fc1)
        # Dense layers
        fcf = Dense(64, activation='relu')(fc1)
        fcf = Dropout(0.25)(fcf)
        fcf = BatchNormalization()(fcf)
    elif dense_lays == 3:  
        # Dense layers
        fc1 = Dense(256, activation='relu')(concat)
        fc1 = Dropout(0.25)(fc1)
        fc1 = BatchNormalization()(fc1)
        # Dense layers
        fc2 = Dense(128, activation='relu')(fc1)
        fc2 = Dropout(0.25)(fc2)
        fc2 = BatchNormalization()(fc2)
        # Dense layers
        fcf = Dense(64, activation='relu')(fc2)
        fcf = Dropout(0.25)(fcf)
        fcf = BatchNormalization()(fcf)
    else:
        print('Number of dense layers must be between 2 and 3')
  
    output = Dense(1, activation='sigmoid')(fcf)


    model = tensorflow.keras.Model(inputs=[int_sequences_input, cons_sequences_input, secs_sequences_input] , outputs=output)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adamax')

    # Print model summary
    print(model.summary())
    return model

def build_params_seq_model(gru_units=64, 
                    dense_lays=2,
                    emb=16,
                    words=16,
                    length_long_seq=150):
    # Define input tensor
    int_sequences_input = Input(shape=(length_long_seq,))
    # Word embedding layer
    embedded_inputs = Embedding(input_dim=words, output_dim=emb, input_length=length_long_seq)(int_sequences_input)

    # Apply dropout to prevent overfitting
    embedded_inputs = SpatialDropout1D(0.25)(embedded_inputs)

    # Apply Conv1D
    conv1d = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(embedded_inputs)
    conv1d = Dropout(0.25)(conv1d)
    conv1d = BatchNormalization()(conv1d)

    # Apply Bidirectional GRU over embedded inputs
    rnn_outs = Bidirectional(GRU(gru_units, return_sequences=True))(conv1d)
    rnn_outs = Dropout(0.25)(rnn_outs) # Apply dropout to GRU outputs to prevent overfitting

    # Attention Mechanism - Generate attention vectors
    seq, word_scores = Attention(return_attention=True, name = "att")(rnn_outs)

    if dense_lays == 2:
        # Dense layers
        fc1 = Dense(128, activation='relu')(seq)
        fc1 = Dropout(0.25)(fc1)
        fc1 = BatchNormalization()(fc1)
        # Dense layers
        fcf = Dense(64, activation='relu')(fc1)
        fcf = Dropout(0.25)(fcf)
        fcf = BatchNormalization()(fcf)
    elif dense_lays == 3:  
        # Dense layers
        fc1 = Dense(256, activation='relu')(seq)
        fc1 = Dropout(0.25)(fc1)
        fc1 = BatchNormalization()(fc1)
        # Dense layers
        fc2 = Dense(128, activation='relu')(fc1)
        fc2 = Dropout(0.25)(fc2)
        fc2 = BatchNormalization()(fc2)
        # Dense layers
        fcf = Dense(64, activation='relu')(fc2)
        fcf = Dropout(0.25)(fcf)
        fcf = BatchNormalization()(fcf)
    else:
        print('Number of dense layers must be between 2 and 3')
    output = Dense(1, activation='sigmoid')(fcf)

    model = tensorflow.keras.Model(inputs=int_sequences_input, outputs=output)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adamax')

    # Print model summary
    print(model.summary())
    return model


def build_params_cons_model(cons_tr, 
                    n_feature_maps=32, 
                    dense_lays=2,
                    resnet_blocks=2):
 
    input_shape_cons = (cons_tr.shape[1], cons_tr.shape[2])
    cons_sequences_input = Input(input_shape_cons)

    # BLOCK 1

    conv_x = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(cons_sequences_input)
    conv_x = Dropout(.25)(conv_x)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = Dropout(.25)(conv_y)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = Dropout(.25)(conv_z)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(cons_sequences_input)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_1 = tensorflow.keras.layers.add([shortcut_y, conv_z])
    output_block_1 = Activation('relu')(output_block_1)

    if resnet_blocks == 2:

        # BLOCK 2

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_final = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_final = Activation('relu')(output_block_final)

    elif resnet_blocks == 3:
  
        # BLOCK 2

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_2 = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_2 = Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = BatchNormalization()(output_block_2)

        output_block_final = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_final = Activation('relu')(output_block_final)

    elif resnet_blocks == 4:
  
        # BLOCK 2

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_2 = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_2 = Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_2)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_3 = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_3 = Activation('relu')(output_block_3)

        # BLOCK 4

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_3)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = BatchNormalization()(output_block_3)

        output_block_final = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_final = Activation('relu')(output_block_final)
    
    else:
        print('Number of resnet blocks must be between 2 and 4')

    # FINAL
    gap_layer = GlobalAveragePooling1D()(output_block_final)

    # model.add(Flatten())
    flatten_1 = Flatten()(gap_layer)

    if dense_lays == 2:
        # Dense layers
        fc1 = Dense(128, activation='relu')(flatten_1)
        fc1 = Dropout(0.25)(fc1)
        fc1 = BatchNormalization()(fc1)
        # Dense layers
        fcf = Dense(64, activation='relu')(fc1)
        fcf = Dropout(0.25)(fcf)
        fcf = BatchNormalization()(fcf)
    elif dense_lays == 3:  
        # Dense layers
        fc1 = Dense(256, activation='relu')(flatten_1)
        fc1 = Dropout(0.25)(fc1)
        fc1 = BatchNormalization()(fc1)
        # Dense layers
        fc2 = Dense(128, activation='relu')(fc1)
        fc2 = Dropout(0.25)(fc2)
        fc2 = BatchNormalization()(fc2)
        # Dense layers
        fcf = Dense(64, activation='relu')(fc2)
        fcf = Dropout(0.25)(fcf)
        fcf = BatchNormalization()(fcf)
    else:
        print('Number of dense layers must be between 2 and 3')
    output = Dense(1, activation='sigmoid')(fcf)


    model = tensorflow.keras.Model(inputs=cons_sequences_input , outputs=output)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adamax')

    # Print model summary
    print(model.summary())
    return model


def build_params_sec_model(secs_tr, 
                    n_feature_maps=32, 
                    dense_lays=2,
                    resnet_blocks=2):

    input_shape_sec = (secs_tr.shape[1], secs_tr.shape[2])
    secs_sequences_input = Input(input_shape_sec)

    # BLOCK 1

    conv_i = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(secs_sequences_input)
    conv_i = Dropout(.25)(conv_i)
    conv_i = BatchNormalization()(conv_i)
    conv_i = Activation('relu')(conv_i)

    conv_j = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_i)
    conv_j = Dropout(.25)(conv_j)
    conv_j = BatchNormalization()(conv_j)
    conv_j = Activation('relu')(conv_j)

    conv_k = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_j)
    conv_k = Dropout(.25)(conv_k)
    conv_k = BatchNormalization()(conv_k)

    # expand channels for the sum
    shortcut_u = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(secs_sequences_input)
    shortcut_u = BatchNormalization()(shortcut_u)

    output_block_1_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
    output_block_1_s = Activation('relu')(output_block_1_s)

    if resnet_blocks == 2:

        # BLOCK 2

        conv_i = Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # expand channels for the sum
        shortcut_u = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1_s)
        shortcut_u = BatchNormalization()(shortcut_u)

        output_block_final_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_final_s = Activation('relu')(output_block_final_s)

    elif resnet_blocks == 3:
        # BLOCK 2

        conv_i = Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # expand channels for the sum
        shortcut_u = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1_s)
        shortcut_u = BatchNormalization()(shortcut_u)

        output_block_2_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_2_s = Activation('relu')(output_block_2_s)

        # BLOCK 3

        conv_i = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # no need to expand channels because they are equal
        shortcut_u = BatchNormalization()(output_block_2_s)

        output_block_final_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_final_s = Activation('relu')(output_block_final_s)

    elif resnet_blocks == 4:
        # BLOCK 2

        conv_i = Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # expand channels for the sum
        shortcut_u = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1_s)
        shortcut_u = BatchNormalization()(shortcut_u)

        output_block_2_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_2_s = Activation('relu')(output_block_2_s)

        # BLOCK 3

        conv_i = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # no need to expand channels because they are equal
        shortcut_u = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_2_s)
        shortcut_u = BatchNormalization()(shortcut_u)

        output_block_3_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_3_s = Activation('relu')(output_block_3_s)

        # BLOCK 4

        conv_i = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_3_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # no need to expand channels because they are equal
        shortcut_u = BatchNormalization()(output_block_3_s)

        output_block_final_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_final_s = Activation('relu')(output_block_final_s)


    # FINAL
    gap_layer_s = GlobalAveragePooling1D()(output_block_final_s)

    # model.add(Flatten())
    flatten_2 = Flatten()(gap_layer_s)

    if dense_lays == 2:
        # Dense layers
        fc1 = Dense(128, activation='relu')(flatten_2)
        fc1 = Dropout(0.25)(fc1)
        fc1 = BatchNormalization()(fc1)
        # Dense layers
        fcf = Dense(64, activation='relu')(fc1)
        fcf = Dropout(0.25)(fcf)
        fcf = BatchNormalization()(fcf)
    elif dense_lays == 3:  
        # Dense layers
        fc1 = Dense(256, activation='relu')(flatten_2)
        fc1 = Dropout(0.25)(fc1)
        fc1 = BatchNormalization()(fc1)
        # Dense layers
        fc2 = Dense(128, activation='relu')(fc1)
        fc2 = Dropout(0.25)(fc2)
        fc2 = BatchNormalization()(fc2)
        # Dense layers
        fcf = Dense(64, activation='relu')(fc2)
        fcf = Dropout(0.25)(fcf)
        fcf = BatchNormalization()(fcf)
    else:
        print('Number of dense layers must be between 2 and 3')
    output = Dense(1, activation='sigmoid')(fcf)


    model = tensorflow.keras.Model(inputs=secs_sequences_input, outputs=output)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adamax')

    # Print model summary
    print(model.summary())
    return model


def build_params_seq_cons_model(cons_tr, 
                    n_feature_maps=32, 
                    dense_lays=2,
                    emb=16,
                    resnet_blocks=2,
                    words=16,
                    length_long_seq=150):
    # Define input tensor
    # inp = Input(shape=(X_train.shape[1],), dtype='int32')
    int_sequences_input = Input(shape=(length_long_seq,))
    # Word embedding layer
    # embedded_inputs = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable = True)(inp)
    embedded_inputs = Embedding(input_dim=words, output_dim=emb, input_length=length_long_seq)(int_sequences_input)

    # Apply dropout to prevent overfitting
    embedded_inputs = SpatialDropout1D(0.25)(embedded_inputs)

    # Apply Conv1D
    conv1d = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(embedded_inputs)
    conv1d = Dropout(0.25)(conv1d)
    conv1d = BatchNormalization()(conv1d)

    # Apply Bidirectional GRU over embedded inputs
    rnn_outs = Bidirectional(GRU(256, return_sequences=True))(conv1d)
    rnn_outs = Dropout(0.25)(rnn_outs) # Apply dropout to GRU outputs to prevent overfitting

    # Attention Mechanism - Generate attention vectors
    seq, word_scores = Attention(return_attention=True, name = "att")(rnn_outs)


    input_shape_cons = (cons_tr.shape[1], cons_tr.shape[2])
    cons_sequences_input = Input(input_shape_cons)

    # BLOCK 1

    conv_x = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(cons_sequences_input)
    conv_x = Dropout(.25)(conv_x)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = Dropout(.25)(conv_y)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = Dropout(.25)(conv_z)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(cons_sequences_input)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_1 = tensorflow.keras.layers.add([shortcut_y, conv_z])
    output_block_1 = Activation('relu')(output_block_1)

    if resnet_blocks == 2:

        # BLOCK 2

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_final = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_final = Activation('relu')(output_block_final)

    elif resnet_blocks == 3:
  
        # BLOCK 2

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_2 = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_2 = Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = BatchNormalization()(output_block_2)

        output_block_final = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_final = Activation('relu')(output_block_final)

    elif resnet_blocks == 4:
  
        # BLOCK 2

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_2 = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_2 = Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_2)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_3 = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_3 = Activation('relu')(output_block_3)

        # BLOCK 4

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_3)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = BatchNormalization()(output_block_3)

        output_block_final = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_final = Activation('relu')(output_block_final)
    
    else:
        print('Number of resnet blocks must be between 2 and 4')

    # FINAL
    gap_layer = GlobalAveragePooling1D()(output_block_final)

    # model.add(Flatten())
    flatten_1 = Flatten()(gap_layer)

    concat = concatenate([seq, flatten_1])

    if dense_lays == 2:
        # Dense layers
        fc1 = Dense(128, activation='relu')(concat)
        fc1 = Dropout(0.25)(fc1)
        fc1 = BatchNormalization()(fc1)
        # Dense layers
        fcf = Dense(64, activation='relu')(fc1)
        fcf = Dropout(0.25)(fcf)
        fcf = BatchNormalization()(fcf)
    elif dense_lays == 3:  
        # Dense layers
        fc1 = Dense(256, activation='relu')(concat)
        fc1 = Dropout(0.25)(fc1)
        fc1 = BatchNormalization()(fc1)
        # Dense layers
        fc2 = Dense(128, activation='relu')(fc1)
        fc2 = Dropout(0.25)(fc2)
        fc2 = BatchNormalization()(fc2)
        # Dense layers
        fcf = Dense(64, activation='relu')(fc2)
        fcf = Dropout(0.25)(fcf)
        fcf = BatchNormalization()(fcf)
    else:
        print('Number of dense layers must be between 2 and 3')
    output = Dense(1, activation='sigmoid')(fcf)


    model = tensorflow.keras.Model(inputs=[int_sequences_input, cons_sequences_input] , outputs=output)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adamax')

    # Print model summary
    print(model.summary())
    return model


def build_params_seq_sec_model(secs_tr, 
                    n_feature_maps=32, 
                    gru_units=64, 
                    dense_lays=2,
                    emb=16,
                    resnet_blocks=2,
                    words=16,
                    length_long_seq=150):
    # Define input tensor
    # inp = Input(shape=(X_train.shape[1],), dtype='int32')
    int_sequences_input = Input(shape=(length_long_seq,))
    # Word embedding layer
    # embedded_inputs = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable = True)(inp)
    embedded_inputs = Embedding(input_dim=words, output_dim=emb, input_length=length_long_seq)(int_sequences_input)

    # Apply dropout to prevent overfitting
    embedded_inputs = SpatialDropout1D(0.25)(embedded_inputs)

    # Apply Conv1D
    conv1d = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(embedded_inputs)
    conv1d = Dropout(0.25)(conv1d)
    conv1d = BatchNormalization()(conv1d)

    # Apply Bidirectional GRU over embedded inputs
    rnn_outs = Bidirectional(GRU(gru_units, return_sequences=True))(conv1d)
    rnn_outs = Dropout(0.25)(rnn_outs) # Apply dropout to GRU outputs to prevent overfitting

    # Attention Mechanism - Generate attention vectors
    seq, word_scores = Attention(return_attention=True, name = "att")(rnn_outs)

    input_shape_sec = (secs_tr.shape[1], secs_tr.shape[2])
    secs_sequences_input = Input(input_shape_sec)

    # BLOCK 1

    conv_i = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(secs_sequences_input)
    conv_i = Dropout(.25)(conv_i)
    conv_i = BatchNormalization()(conv_i)
    conv_i = Activation('relu')(conv_i)

    conv_j = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_i)
    conv_j = Dropout(.25)(conv_j)
    conv_j = BatchNormalization()(conv_j)
    conv_j = Activation('relu')(conv_j)

    conv_k = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_j)
    conv_k = Dropout(.25)(conv_k)
    conv_k = BatchNormalization()(conv_k)

    # expand channels for the sum
    shortcut_u = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(secs_sequences_input)
    shortcut_u = BatchNormalization()(shortcut_u)

    output_block_1_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
    output_block_1_s = Activation('relu')(output_block_1_s)

    if resnet_blocks == 2:

        # BLOCK 2

        conv_i = Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # expand channels for the sum
        shortcut_u = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1_s)
        shortcut_u = BatchNormalization()(shortcut_u)

        output_block_final_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_final_s = Activation('relu')(output_block_final_s)

    elif resnet_blocks == 3:
        # BLOCK 2

        conv_i = Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # expand channels for the sum
        shortcut_u = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1_s)
        shortcut_u = BatchNormalization()(shortcut_u)

        output_block_2_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_2_s = Activation('relu')(output_block_2_s)

        # BLOCK 3

        conv_i = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # no need to expand channels because they are equal
        shortcut_u = BatchNormalization()(output_block_2_s)

        output_block_final_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_final_s = Activation('relu')(output_block_final_s)

    elif resnet_blocks == 4:
        # BLOCK 2

        conv_i = Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # expand channels for the sum
        shortcut_u = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1_s)
        shortcut_u = BatchNormalization()(shortcut_u)

        output_block_2_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_2_s = Activation('relu')(output_block_2_s)

        # BLOCK 3

        conv_i = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # no need to expand channels because they are equal
        shortcut_u = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_2_s)
        shortcut_u = BatchNormalization()(shortcut_u)

        output_block_3_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_3_s = Activation('relu')(output_block_3_s)

        # BLOCK 4

        conv_i = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_3_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # no need to expand channels because they are equal
        shortcut_u = BatchNormalization()(output_block_3_s)

        output_block_final_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_final_s = Activation('relu')(output_block_final_s)


    # FINAL
    gap_layer_s = GlobalAveragePooling1D()(output_block_final_s)

    # model.add(Flatten())
    flatten_2 = Flatten()(gap_layer_s)

    concat = concatenate([seq, flatten_2])

    if dense_lays == 2:
        # Dense layers
        fc1 = Dense(128, activation='relu')(concat)
        fc1 = Dropout(0.25)(fc1)
        fc1 = BatchNormalization()(fc1)
        # Dense layers
        fcf = Dense(64, activation='relu')(fc1)
        fcf = Dropout(0.25)(fcf)
        fcf = BatchNormalization()(fcf)
    elif dense_lays == 3:  
        # Dense layers
        fc1 = Dense(256, activation='relu')(concat)
        fc1 = Dropout(0.25)(fc1)
        fc1 = BatchNormalization()(fc1)
        # Dense layers
        fc2 = Dense(128, activation='relu')(fc1)
        fc2 = Dropout(0.25)(fc2)
        fc2 = BatchNormalization()(fc2)
        # Dense layers
        fcf = Dense(64, activation='relu')(fc2)
        fcf = Dropout(0.25)(fcf)
        fcf = BatchNormalization()(fcf)
    else:
        print('Number of dense layers must be between 2 and 3')
    output = Dense(1, activation='sigmoid')(fcf)


    model = tensorflow.keras.Model(inputs=[int_sequences_input, secs_sequences_input] , outputs=output)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adamax')

    # Print model summary
    print(model.summary())
    return model


def build_params_cons_sec_model(cons_tr, 
                    secs_tr, 
                    n_feature_maps=32, 
                    dense_lays=2,
                    resnet_blocks=2):

    input_shape_cons = (cons_tr.shape[1], cons_tr.shape[2])
    cons_sequences_input = Input(input_shape_cons)

    # BLOCK 1

    conv_x = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(cons_sequences_input)
    conv_x = Dropout(.25)(conv_x)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = Dropout(.25)(conv_y)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = Dropout(.25)(conv_z)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(cons_sequences_input)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_1 = tensorflow.keras.layers.add([shortcut_y, conv_z])
    output_block_1 = Activation('relu')(output_block_1)

    if resnet_blocks == 2:

        # BLOCK 2

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_final = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_final = Activation('relu')(output_block_final)

    elif resnet_blocks == 3:
  
        # BLOCK 2

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_2 = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_2 = Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = BatchNormalization()(output_block_2)

        output_block_final = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_final = Activation('relu')(output_block_final)

    elif resnet_blocks == 4:
  
        # BLOCK 2

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_2 = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_2 = Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        conv_z = BatchNormalization()(conv_z)

        shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_2)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_3 = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_3 = Activation('relu')(output_block_3)

        # BLOCK 4

        conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_3)
        conv_x = Dropout(.25)(conv_x)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = Dropout(.25)(conv_y)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = Dropout(.25)(conv_z)
        output_block_final = tensorflow.keras.layers.add([shortcut_y, conv_z])
        output_block_final = Activation('relu')(output_block_final)
    
    else:
        print('Number of resnet blocks must be between 2 and 4')

    # FINAL
    gap_layer = GlobalAveragePooling1D()(output_block_final)

    # model.add(Flatten())
    flatten_1 = Flatten()(gap_layer)


    input_shape_sec = (secs_tr.shape[1], secs_tr.shape[2])
    secs_sequences_input = Input(input_shape_sec)

    # BLOCK 1

    conv_i = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(secs_sequences_input)
    conv_i = Dropout(.25)(conv_i)
    conv_i = BatchNormalization()(conv_i)
    conv_i = Activation('relu')(conv_i)

    conv_j = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_i)
    conv_j = Dropout(.25)(conv_j)
    conv_j = BatchNormalization()(conv_j)
    conv_j = Activation('relu')(conv_j)

    conv_k = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_j)
    conv_k = Dropout(.25)(conv_k)
    conv_k = BatchNormalization()(conv_k)

    # expand channels for the sum
    shortcut_u = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(secs_sequences_input)
    shortcut_u = BatchNormalization()(shortcut_u)

    output_block_1_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
    output_block_1_s = Activation('relu')(output_block_1_s)

    if resnet_blocks == 2:

        # BLOCK 2

        conv_i = Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # expand channels for the sum
        shortcut_u = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1_s)
        shortcut_u = BatchNormalization()(shortcut_u)

        output_block_final_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_final_s = Activation('relu')(output_block_final_s)

    elif resnet_blocks == 3:
        # BLOCK 2

        conv_i = Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # expand channels for the sum
        shortcut_u = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1_s)
        shortcut_u = BatchNormalization()(shortcut_u)

        output_block_2_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_2_s = Activation('relu')(output_block_2_s)

        # BLOCK 3

        conv_i = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # no need to expand channels because they are equal
        shortcut_u = BatchNormalization()(output_block_2_s)

        output_block_final_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_final_s = Activation('relu')(output_block_final_s)

    elif resnet_blocks == 4:
        # BLOCK 2

        conv_i = Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # expand channels for the sum
        shortcut_u = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1_s)
        shortcut_u = BatchNormalization()(shortcut_u)

        output_block_2_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_2_s = Activation('relu')(output_block_2_s)

        # BLOCK 3

        conv_i = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # no need to expand channels because they are equal
        shortcut_u = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_2_s)
        shortcut_u = BatchNormalization()(shortcut_u)

        output_block_3_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_3_s = Activation('relu')(output_block_3_s)

        # BLOCK 4

        conv_i = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_3_s)
        conv_i = Dropout(.25)(conv_i)
        conv_i = BatchNormalization()(conv_i)
        conv_i = Activation('relu')(conv_i)

        conv_j = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_i)
        conv_j = Dropout(.25)(conv_j)
        conv_j = BatchNormalization()(conv_j)
        conv_j = Activation('relu')(conv_j)

        conv_k = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_j)
        conv_k = Dropout(.25)(conv_k)
        conv_k = BatchNormalization()(conv_k)

        # no need to expand channels because they are equal
        shortcut_u = BatchNormalization()(output_block_3_s)

        output_block_final_s = tensorflow.keras.layers.add([shortcut_u, conv_k])
        output_block_final_s = Activation('relu')(output_block_final_s)

    # FINAL
    gap_layer_s = GlobalAveragePooling1D()(output_block_final_s)

    # model.add(Flatten())
    flatten_2 = Flatten()(gap_layer_s)

    concat = concatenate([flatten_1, flatten_2])

    if dense_lays == 2:
        # Dense layers
        fc1 = Dense(128, activation='relu')(concat)
        fc1 = Dropout(0.25)(fc1)
        fc1 = BatchNormalization()(fc1)
        # Dense layers
        fcf = Dense(64, activation='relu')(fc1)
        fcf = Dropout(0.25)(fcf)
        fcf = BatchNormalization()(fcf)
    elif dense_lays == 3:  
        # Dense layers
        fc1 = Dense(256, activation='relu')(concat)
        fc1 = Dropout(0.25)(fc1)
        fc1 = BatchNormalization()(fc1)
        # Dense layers
        fc2 = Dense(128, activation='relu')(fc1)
        fc2 = Dropout(0.25)(fc2)
        fc2 = BatchNormalization()(fc2)
        # Dense layers
        fcf = Dense(64, activation='relu')(fc2)
        fcf = Dropout(0.25)(fcf)
        fcf = BatchNormalization()(fcf)
    else:
        print('Number of dense layers must be between 2 and 3')
    output = Dense(1, activation='sigmoid')(fcf)


    model = tensorflow.keras.Model(inputs=[cons_sequences_input, secs_sequences_input] , outputs=output)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adamax')

    # Print model summary
    print(model.summary())
    return model