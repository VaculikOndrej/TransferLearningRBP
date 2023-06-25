# Transfer Learning RBP

This repo allows to pre-train a baseline model on the binary data composed of positive and negative binding sites of various RNA binding proteins (RBP), and then uses the gained knowledge to fine-tune it on individual protein datasets to get an optimal performance, especially for proteins with insufficient amount of training data. 

Currently, users must use our data or prepare their own in the following **.tsv** form:

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

`/data` folder contains all the tokenizers needed, other data may be downloaded from the following link: LINK

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
The data used in our project may be downloaded from: **TO BE DONE**

## Citing Transfer Learning RBP
If you use Transfer Learning RBP in your research, please cite it as follows:

**TO BE DONE**
