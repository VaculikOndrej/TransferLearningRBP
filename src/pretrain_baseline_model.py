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
from sklearn.model_selection import train_test_split
import tensorflow 
import tensorflow.keras
import tensorflow.keras.backend as K

from utils.callbacks import EarlyStopping_callback
from utils.data_loader import load_and_prepare
from utils.models import build_params_complete_model as compl_model


sns.set()

tensorflow.random.set_seed(73)
seed(73)
seed = 73

LENGTH_LONG_SEQ = 150
REDUCTION = 0.1
BATCH = 256

#################################################################################################################################
#################################################################################################################################
# name of the dataset
DATASET = 'rbp24'

# path where the final baseline model will be stored
OUTPUT_MODEL_PATH = Path(f'PATH/TO/trained_models/{DATASET}/test/')

# path to the training binary data
PATH_TO_TRAINING_DATA = Path('PATH/TO/train.tsv')

# path to the testing binary data
PATH_TO_TEST_DATA = Path('PATH/TO/dev.tsv')

# path to the directory containing all tokenizers
PATH_TO_TOKENIZERS = Path('/data/tokenizers/')

# path to the parameters that were defined by the baseline model hyperparameter tuning process as the optimal ones
PATH_TO_PRECALC_PARAMS = Path('PATH/TO/optimalParams.tsv')
#################################################################################################################################
#################################################################################################################################
precalcParamsDf = pd.read_csv(PATH_TO_PRECALC_PARAMS, sep='\t', usecols=['parameter', 'value'])
precalsParamsDict = pd.Series(precalcParamsDf.value.values, index=precalcParamsDf.parameter).to_dict()
EMBED = precalsParamsDict['embed']
GRU = precalsParamsDict['gru']
TOK = precalsParamsDict['tok']
RESBL = precalsParamsDict['resBl']
FEAT = precalsParamsDict['feat']
DENSE = precalsParamsDict['dense']

OPTIM_TOKENIZER = PATH_TO_TOKENIZERS / f'transcriptome_hg19_{TOK}words_bpe.tokenizer.json'

padded_sequences_train, cons_train, secs_train, y_train_raw = load_and_prepare(PATH_TO_TRAINING_DATA, OPTIM_TOKENIZER, REDUCTION, LENGTH_LONG_SEQ)
X_train, X_test, X_train_cons, X_test_cons, X_train_secs, X_test_secs, y_train, y_test = train_test_split(padded_sequences_train, cons_train, secs_train, y_train_raw, test_size=.1, random_state=seed)

earlystop_callback = EarlyStopping_callback()

model = compl_model(X_train_cons, X_train_secs, 
                    n_feature_maps=FEAT, 
                    gru_units=GRU, 
                    resnet_blocks=RESBL,
                    dense_lays=DENSE, 
                    emb=EMBED, 
                    words=TOK)
K.set_value(model.optimizer.learning_rate, 0.001)
hist = model.fit([X_train, X_train_cons, X_train_secs], y_train, validation_data=([X_test, X_test_cons, X_test_secs], y_test), 
                epochs=50, batch_size=BATCH, callbacks=[earlystop_callback])
loss = hist.history['loss'];acc = hist.history['accuracy']
val_loss = hist.history['val_loss'];val_acc = hist.history['val_accuracy']
val_pred = model.predict([X_test, X_test_cons, X_test_secs], batch_size=BATCH, verbose=1)
auc = roc_auc_score(y_test, val_pred)

model_output = OUTPUT_MODEL_PATH / 'final_basemodel.h5'
auc_score_output = OUTPUT_MODEL_PATH / 'final_basemodel.txt'
model.save(model_output)
with open(auc_score_output, 'a') as out:
    out.write(f'final basemodel auc score: {auc}\n')
