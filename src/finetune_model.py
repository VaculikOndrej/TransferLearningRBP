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
from tensorflow.keras.models import load_model

from utils.att_layer import Attention 
from utils.callbacks import EarlyStopping_callback
from utils.data_loader import load_and_prepare


sns.set()

tensorflow.random.set_seed(73)
seed(73)
seed = 73

LENGTH_LONG_SEQ = 150
BATCH = 256

###########################################################################################################################################
###########################################################################################################################################
# name of dataset
DATASET = 'rbp24'

# path to pretrained basemodel
BASEMODEL_PATH = Path('PATH/TO/final_basemodel.h5')

# path where the final models and their data will be saved
OUTPUT_MODEL_PATH = Path(f'PATH/TO/trained_models/{DATASET}/test/finetuned/')

# path to the directory with all the individual dataset
PATH_TO_DATA = Path('PATH/TO/final_datasets/rbp24_prots_sec_1_to_1')
# path to the tokenizer that was chosen as the best during the basemodel hyperparameter tuning process
PATH_TO_TOKENIZERS = Path('/data/tokenizers/')

# path to the optimal parameters defined during the basemodel hyperparameter tuning process
PATH_TO_PRECALC_PARAMS = Path('PATH/TO/optimalParams.tsv')
###########################################################################################################################################
###########################################################################################################################################
precalcParamsDf = pd.read_csv(PATH_TO_PRECALC_PARAMS, sep='\t', usecols=['parameter', 'value'])
precalsParamsDict = pd.Series(precalcParamsDf.value.values, index=precalcParamsDf.parameter).to_dict()
TOK = precalsParamsDict['tok']
OPTIM_TOKENIZER = str(PATH_TO_TOKENIZERS / f'transcriptome_hg19_{TOK}words_bpe.tokenizer.json')

prots_to_train = [os.path.basename(f) for f in os.scandir(PATH_TO_DATA) if f.is_dir()]

model_name = BASEMODEL_PATH.stem
earlystop_callback = EarlyStopping_callback()

for prot in prots_to_train:
    print(f'\nStarting training process on {prot} with model: {model_name}... \n')

    output_prot_models_path = OUTPUT_MODEL_PATH / prot
    output_prot_models_path.mkdir(exist_ok=True, parents=True)

    train_prot = PATH_TO_DATA / f'{prot}/train.tsv'
    eval_prot = PATH_TO_DATA / f'{prot}/dev.tsv'

    padded_sequences_train, cons_train, secs_train, y_train_raw = load_and_prepare(train_prot, OPTIM_TOKENIZER, LENGTH_LONG_SEQ)
    X_train, X_val, X_train_cons, X_val_cons, X_train_secs, X_val_secs, y_train, y_val = train_test_split(padded_sequences_train, cons_train, secs_train, y_train_raw, test_size=.1, random_state=seed)
    X_test, X_test_cons, X_test_secs, y_test = load_and_prepare(eval_prot, OPTIM_TOKENIZER, LENGTH_LONG_SEQ)

    model = load_model(BASEMODEL_PATH, custom_objects={'Attention': Attention})
    tensorflow.keras.backend.set_value(model.optimizer.lr, 0.001)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='Adamax')
    hist = model.fit([X_train, X_train_cons, X_train_secs], y_train, validation_data=([X_val, X_val_cons, X_val_secs], y_val), 
                    epochs=50, batch_size=128, callbacks=[earlystop_callback], verbose=1)
    loss = hist.history['loss']; acc = hist.history['accuracy']
    val_loss = hist.history['val_loss']; val_acc = hist.history['val_accuracy']

    eval_pred = model.predict([X_test, X_test_cons, X_test_secs], batch_size=128, verbose=1)

    auc = roc_auc_score(y_test, eval_pred)
    print(f'\nModel: {model_name} finetuned on {prot} achieved following AUC score: {auc} \n')
    
    model_outp = output_prot_models_path / f'finetuned_model_{prot}.h5'
    model.save(output_prot_models_path / f'finetuned_model_{prot}.h5')

    with open(output_prot_models_path / f'achieved_AUC_{prot}.txt', 'a') as eval_tab:
        eval_tab.write(f'{prot}\t{model_outp.stem}\t{auc}\n')

print(f'\nTraining process on {DATASET} dataset successfully completed... \n')
