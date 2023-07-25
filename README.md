# Transfer Learning RBP

Link to the related scientific work: **TO BE ADDED**

This repository allows to pre-train a baseline model on binary data containing positive and negative binding sites of various RNA binding proteins (RBP). The resulting model can be further fine-tuned on individual protein datasets to achieve an optimal performance, particularly benefiting proteins with limited training data.

Currently, users have the option to utilize our provided data or prepare their own datasets using the following **.tsv** format:

`header RNASequence  EvolConservation  secStruct  label`

where,
 - **header** = chr3:47114069-47114219(-)
 - **RNASequence** = TTTCAGATTCTTTAA...
 - **EvolConservation** = [0.01, 0.032, 0.044, 0.041, ..., 0.003]
   - array of conservation scores
 - **RNASecondStructure** = (((((((....))...
   - RNA secondary structure sequence generated by ViennaRNA2 package (https://www.tbi.univie.ac.at/RNA/)
 - **label** = 0 or 1
   - 0 = negative, 1 = positive

## Data
`/data/tokenizers` folder contains all the tokenizers needed.

`/data/training_data` folder contains all the training and evaluation data used in our project.

**NOTE:** The following data are compressed due to their size and must be decompressed before users use them:
```
/data/training_data/PTB/train.tar.gz
/data/training_data/rbp24_binary/train.tar.gz
/data/training_data/rbp24_proteins/CLIPSEQ_AGO2/train.tar.gz
/data/training_data/rbp24_proteins/ICLIP_TDP43/train.tar.gz
/data/training_data/rbp24_proteins/ICLIP_TIAL1/train.tar.gz
/data/training_data/rbp24_proteins/PARCLIP_HUR/train.tar.gz
/data/training_data/rbp31_binary/train.tar.gz
```
## Code
`/src` folder contains four individual scripts, which can run individually, however user must run them in the following order:
  1. `basemodel_parameter_search.py`
  2. `pretrain_baseline_model.py`
  3. `finetune_model.py`
  4. `visualize_important_kmers.ipynb`
     
To run the scripts, users must clone the repo and edit every script as there are Path variables to individual data, which must be changed to proper Paths. Apart from these variables, there are modifiable variables such as sequence length (LENGTH_LONG_SEQ), batch size (BATCH), or number of StratifiedKFold splits (SPLITS).

## Usage
If you want to run experiments from this repository, use the following commands to clone the repository and install the environmental requirements into the virtual environment.
Users must have installed Conda (https://docs.conda.io/en/latest/) to set the correct environment to run the scripts.
```
git clone https://github.com/ondravaculik/TransferLearningRBP.git
cd TransferLearningRBP

conda env create -n ENVNAME --file /environment/environment.yml

pip install -r environment/requirements.txt
```

## Citing Transfer Learning RBP
If you use Transfer Learning RBP in your research, please cite it as follows:

**TO BE DONE**
