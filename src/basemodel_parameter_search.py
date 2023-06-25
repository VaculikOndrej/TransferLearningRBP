from IPython.core.display import HTML
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy.random import seed
import os
import pandas as pd
from pathlib import Path
import random
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import tensorflow 
import tensorflow.keras
import tensorflow.keras.backend as K

from utils.att_layer import Attention 
from utils.callbacks import EarlyStopping_callback
from utils.data_loader import load_and_prepare
from utils.models import build_params_complete_model as compl_model
from pretrain.eval_models import get_highest_average


sns.set()

tensorflow.random.set_seed(73)
seed(73)
seed = 73

LENGTH_LONG_SEQ = 150
REDUCTION = 0.1
BATCH = 256
SPLITS = 10

optimParams = []

earlystop_callback = EarlyStopping_callback()

print('Starting fine-tuning process.')

## Baseline model fine-tuning part A
## Embedding dims and GRU units
words = 16

############################################################################################################################################################
############################################################################################################################################################

# path where the final baseline models will be stored
outf_path_dir = Path('pathToOutputDirectory/trained_models/test')
outf_path_dir.mkdir(exist_ok=True, parents=True)

# path to training binary data - train.tsv
train_rbp = Path('pathToTrainingData/train.tsv')

#paths to individual tokenizers
tokenizer_16_path = f'data/tokenizers/transcriptome_hg19_16words_bpe.tokenizer.json'
tokenizer_32_path = f'data/tokenizers/transcriptome_hg19_32words_bpe.tokenizer.json'
tokenizer_64_path = f'data/tokenizers/transcriptome_hg19_64words_bpe.tokenizer.json'
############################################################################################################################################################
############################################################################################################################################################

padded_sequences_train16, cons_train16, secs_train16, y_train_raw16 = load_and_prepare(train_rbp, tokenizer_16_path, REDUCTION, LENGTH_LONG_SEQ)

n_features=[32]
gru_units=[64, 128, 256, 512]
dense_lays=[2]
resnet_blocks=[2]
embed_dims=[16, 32, 64, 128]

dict_of_variants_aucs = {}

for e in embed_dims:
    for u in gru_units:
        name = f'{2}_{2}_{e}_{32}_{u}_{16}'
        dict_of_variants_aucs[name] = []
                
skf = StratifiedKFold(n_splits=SPLITS)
for train_indices, test_indices in skf.split(padded_sequences_train16, y_train_raw16):
    X_train, X_train_cons, X_train_secs, y_train = padded_sequences_train16[train_indices],\
                                cons_train16[train_indices],\
                                secs_train16[train_indices],\
                                y_train_raw16[train_indices]
    X_test, X_test_cons, X_test_secs, y_test = padded_sequences_train16[test_indices],\
                                cons_train16[test_indices],\
                                secs_train16[test_indices],\
                                y_train_raw16[test_indices]
    for e in embed_dims:
        for u in gru_units:
            name = f'{2}_{2}_{e}_{32}_{u}_{16}'

            model = compl_model(X_train_cons, X_train_secs, n_feature_maps=32,
                    gru_units=u, resnet_blocks=2, dense_lays=2, emb=e, words=words)
            K.set_value(model.optimizer.learning_rate, 0.001)
            hist = model.fit([X_train, X_train_cons, X_train_secs], y_train, validation_data=([X_test, X_test_cons, X_test_secs], y_test), 
                          epochs=1, batch_size=BATCH, callbacks=[earlystop_callback])
            val_pred = model.predict([X_test, X_test_cons, X_test_secs], batch_size=BATCH, verbose=1)
            compl_auc = roc_auc_score(y_test, val_pred)
            dict_of_variants_aucs[name].append(compl_auc)

df = pd.DataFrame.from_dict(dict_of_variants_aucs, orient='index').transpose()

outf_path = outf_path_dir / 'sequence_branch_finetuning_A_emb_units_10folds.tsv'
df.to_csv(outf_path, sep='\t')

best_params, _ = get_highest_average(dict_of_variants_aucs)
_, _, final_embed_dims, _, final_gru_units, _ = [int(param) for param in best_params.split('_')]

print('Embedding and BiGru units done!')

optimParams.append(('embed', final_embed_dims))
optimParams.append(('gru', final_gru_units))

## Baseline model fine-tuning part B
## Tokenizer - number of words
padded_sequences_train32, cons_train32, secs_train32, y_train_raw32 = load_and_prepare(train_rbp, tokenizer_32_path, REDUCTION, LENGTH_LONG_SEQ)
padded_sequences_train64, cons_train64, secs_train64, y_train_raw64 = load_and_prepare(train_rbp, tokenizer_64_path, REDUCTION, LENGTH_LONG_SEQ)

n_features = [32]
bigru_units = final_gru_units
dense_lays = [2]
resnet_blocks = [2]
embed_dims = final_embed_dims

tok_words = [16, 32, 64]

dict_of_variants_aucs = {}

for t in tok_words:
    name = f'{2}_{2}_{final_embed_dims}_{32}_{final_gru_units}_{t}'
    dict_of_variants_aucs[name] = []
   
skf = StratifiedKFold(n_splits=SPLITS)
for train_indices, test_indices in skf.split(padded_sequences_train16, y_train_raw16):
    X_train, X_train_cons, X_train_secs, y_train = padded_sequences_train16[train_indices],\
                                cons_train16[train_indices],\
                                secs_train16[train_indices],\
                                y_train_raw16[train_indices]
    X_test, X_test_cons, X_test_secs, y_test = padded_sequences_train16[test_indices],\
                                cons_train16[test_indices],\
                                secs_train16[test_indices],\
                                y_train_raw16[test_indices]
    
    X_train32, X_train_cons32, X_train_secs32, y_train32 = padded_sequences_train32[train_indices],\
                                cons_train32[train_indices],\
                                secs_train32[train_indices],\
                                y_train_raw32[train_indices]
    X_test32, X_test_cons32, X_test_secs32, y_test32 = padded_sequences_train32[test_indices],\
                                cons_train32[test_indices],\
                                secs_train32[test_indices],\
                                y_train_raw32[test_indices]
    
    X_train64, X_train_cons64, X_train_secs64, y_train64 = padded_sequences_train64[train_indices],\
                                cons_train64[train_indices],\
                                secs_train64[train_indices],\
                                y_train_raw64[train_indices]
    X_test64, X_test_cons64, X_test_secs64, y_test64 = padded_sequences_train64[test_indices],\
                                cons_train64[test_indices],\
                                secs_train64[test_indices],\
                                y_train_raw64[test_indices]
    
    model16_name = f'{2}_{2}_{final_embed_dims}_{32}_{final_gru_units}_{16}'
    model16 = compl_model(X_train_cons, X_train_secs, 
                        n_feature_maps=32, gru_units=bigru_units, resnet_blocks=2, dense_lays=2, emb=embed_dims, words=16)
    K.set_value(model16.optimizer.learning_rate, 0.001)
    hist = model16.fit([X_train, X_train_cons, X_train_secs], y_train, validation_data=([X_test, X_test_cons, X_test_secs], y_test), 
                  epochs=1, batch_size=BATCH, callbacks=[earlystop_callback])
    val_pred = model16.predict([X_test, X_test_cons, X_test_secs], batch_size=BATCH, verbose=1)
    auc16 = roc_auc_score(y_test, val_pred)
    dict_of_variants_aucs[model16_name].append(auc16)
    
    model32_name = f'{2}_{2}_{final_embed_dims}_{32}_{final_gru_units}_{32}'
    model32 = compl_model(X_train_cons32, X_train_secs32, 
                        n_feature_maps=32, gru_units=bigru_units, resnet_blocks=2, dense_lays=2, emb=embed_dims, words=32)
    K.set_value(model32.optimizer.learning_rate, 0.001)
    hist = model32.fit([X_train32, X_train_cons32, X_train_secs32], y_train32, validation_data=([X_test32, X_test_cons32, X_test_secs32], y_test32), 
                  epochs=1, batch_size=BATCH, callbacks=[earlystop_callback])
    val_pred32 = model32.predict([X_test32, X_test_cons32, X_test_secs32], batch_size=BATCH, verbose=1)
    auc32 = roc_auc_score(y_test32, val_pred32)
    dict_of_variants_aucs[model32_name].append(auc32)
     
    model64_name = f'{2}_{2}_{final_embed_dims}_{32}_{final_gru_units}_{64}'
    model64 = compl_model(X_train_cons64, X_train_secs64, 
                        n_feature_maps=32, gru_units=bigru_units, resnet_blocks=2, dense_lays=2, emb=embed_dims, words=64)
    K.set_value(model64.optimizer.learning_rate, 0.001)
    hist = model64.fit([X_train64, X_train_cons64, X_train_secs64], y_train64, validation_data=([X_test64, X_test_cons64, X_test_secs64], y_test64), 
                  epochs=1, batch_size=BATCH, callbacks=[earlystop_callback])
    val_pred64 = model64.predict([X_test64, X_test_cons64, X_test_secs64], batch_size=BATCH, verbose=1)
    auc64 = roc_auc_score(y_test64, val_pred64)
    dict_of_variants_aucs[model64_name].append(auc64)

df = pd.DataFrame.from_dict(dict_of_variants_aucs, orient='index').transpose()

outf_path = outf_path_dir / 'sequence_branch_finetuning_B_tokenizer_10_folds.tsv'
df.to_csv(outf_path, sep='\t')

best_params, _ = get_highest_average(dict_of_variants_aucs)
_, _, _, _, _, final_tok_words = [int(param) for param in best_params.split('_')]

optimParams.append(('tok', final_tok_words))

best_tok_path = f'data/tokenizers/transcriptome_hg19_{final_tok_words}words_bpe.tokenizer.json'

del padded_sequences_train16, cons_train16, secs_train16, y_train_raw16
del padded_sequences_train32, cons_train32, secs_train32, y_train_raw32 
del padded_sequences_train64, cons_train64, secs_train64, y_train_raw64 

print('Tokenizer done!')

# Baseline model fine-tuning part C
# Resnet blocks and number of filters in convolutional layers
padded_sequences_train, cons_train, secs_train, y_train_raw = load_and_prepare(train_rbp, best_tok_path, REDUCTION, LENGTH_LONG_SEQ)

n_features = [32, 64, 128]
bigru_units = final_gru_units
dense_lays = [2]
resnet_blocks = [2, 3]
embed_dims = final_embed_dims

dict_of_variants_aucs = {}

for f in n_features:
    for b in resnet_blocks:
        name = f'{b}_{2}_{final_embed_dims}_{f}_{final_gru_units}_{final_tok_words}'
        dict_of_variants_aucs[name] = []
                
skf = StratifiedKFold(n_splits=SPLITS)
for train_indices, test_indices in skf.split(padded_sequences_train, y_train_raw):
    X_train, X_train_cons, X_train_secs, y_train = padded_sequences_train[train_indices],\
                                cons_train[train_indices],\
                                secs_train[train_indices],\
                                y_train_raw[train_indices]
    X_test, X_test_cons, X_test_secs, y_test = padded_sequences_train[test_indices],\
                                cons_train[test_indices],\
                                secs_train[test_indices],\
                                y_train_raw[test_indices]
    for f in n_features:
        for b in resnet_blocks:
            name = f'{b}_{2}_{final_embed_dims}_{f}_{final_gru_units}_{final_tok_words}'

            model = compl_model(X_train_cons, X_train_secs, 
                                n_feature_maps=f, gru_units=bigru_units, resnet_blocks=b, dense_lays=2, emb=embed_dims, words=final_tok_words)
            K.set_value(model.optimizer.learning_rate, 0.001)
            hist = model.fit([X_train, X_train_cons, X_train_secs], y_train, validation_data=([X_test, X_test_cons, X_test_secs], y_test), 
                          epochs=1, batch_size=BATCH, callbacks=[earlystop_callback])
            val_pred = model.predict([X_test, X_test_cons, X_test_secs], batch_size=BATCH, verbose=1)
            compl_auc = roc_auc_score(y_test, val_pred)
            dict_of_variants_aucs[name].append(compl_auc)

df = pd.DataFrame.from_dict(dict_of_variants_aucs, orient='index').transpose()

outf_path = outf_path_dir / 'cons_sec_branches_finetuning_feats_blocks_10_folds.tsv'
df.to_csv(outf_path, sep='\t')

best_params, _ = get_highest_average(dict_of_variants_aucs)
final_blocks, _, _, final_feats, _, _ = [int(param) for param in best_params.split('_')]

optimParams.append(('resBl', final_blocks))
optimParams.append(('feat', final_feats))

print('Blocks and features done!')

## Baseline model fine-tuning part D
## Number of dense layers in the model's common part
n_features = final_feats
bigru_units = final_gru_units
dense_lays = [2, 3]
resnet_blocks = final_blocks
embed_dims = final_embed_dims

dict_of_variants_aucs = {}

for d in dense_lays:
    name = f'{final_blocks}_{d}_{final_embed_dims}_{final_feats}_{final_gru_units}_{final_tok_words}'
    dict_of_variants_aucs[name] = []
                
skf = StratifiedKFold(n_splits=SPLITS)
for train_indices, test_indices in skf.split(padded_sequences_train, y_train_raw):
    X_train, X_train_cons, X_train_secs, y_train = padded_sequences_train[train_indices],\
                                cons_train[train_indices],\
                                secs_train[train_indices],\
                                y_train_raw[train_indices]
    X_test, X_test_cons, X_test_secs, y_test = padded_sequences_train[test_indices],\
                                cons_train[test_indices],\
                                secs_train[test_indices],\
                                y_train_raw[test_indices]
    for d in dense_lays:
        name = f'{final_blocks}_{d}_{final_embed_dims}_{final_feats}_{final_gru_units}_{final_tok_words}'

        model = compl_model(X_train_cons, X_train_secs, 
                            n_feature_maps=n_features, gru_units=bigru_units, resnet_blocks=resnet_blocks, dense_lays=d, emb=embed_dims, words=final_tok_words)
        K.set_value(model.optimizer.learning_rate, 0.001)
        hist = model.fit([X_train, X_train_cons, X_train_secs], y_train, validation_data=([X_test, X_test_cons, X_test_secs], y_test), 
                      epochs=1, batch_size=BATCH, callbacks=[earlystop_callback])
        val_pred = model.predict([X_test, X_test_cons, X_test_secs], batch_size=BATCH, verbose=1)
        compl_auc = roc_auc_score(y_test, val_pred)
        dict_of_variants_aucs[name].append(compl_auc)

df = pd.DataFrame.from_dict(dict_of_variants_aucs, orient='index').transpose()

outf_path = outf_path_dir / 'common_part_finetuning_dense_lays_10_folds.tsv'
df.to_csv(outf_path, sep='\t')

best_params, _ = get_highest_average(dict_of_variants_aucs)
_, final_dense, _, _, _, _ = [int(param) for param in best_params.split('_')]

optimParams.append(('dense', final_dense))

del padded_sequences_train, cons_train, secs_train, y_train_raw

print('Dense layers done!')

optim_params_path = outf_path_dir / 'optimalParams.tsv'
df = pd.DataFrame(optimParams, columns = ['parameter', 'value'])
df.to_csv('./optimalParams.tsv', sep='\t')
