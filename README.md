# LIMUC (Labeled Images for Ulcerative Colitis) Dataset repository

In this repository you will find scripts to process and train LIMUC dataset.

## How to use this repository?

1. Download the LIMUC dataset from 
2. Run the desired script in `dataset/` folder to create train-val-test sets.
3. Run the training script for the target task.


##  Which script in dataset folder should be run?

### 1. If you want to train your model on a single train-val sets and get the performance on test set:

1.1 After downloading the dataset, run `split_train_val_test.py` to create different sets that are splited by patient-level.
1.2 Run
 

