# LIMUC (Labeled Images for Ulcerative Colitis) Dataset repository

In this repository you will find scripts to process and train LIMUC dataset.

## How to use this repository?

1. Download the LIMUC dataset from [here]().
2. Unzip downloaded files.
3. Set up [wandb](https://wandb.ai/) platform on your machine.
4. Run the desired script in `dataset/` folder to create train-val-test sets for the target task (see below).
5. Run the training script for the target task.


##  Which script in the dataset folder should be run?

### 1. If you want to train your model on a single train-val sets and get the performance on test set:

1.1 After downloading the dataset, run `split_train_val_test.py` on `patient_based_classified_images` to create different sets that are splited by patient-level.  
Example: 
```
python split_train_val_test.py 
--target_dir "path to target folder where train, val, and test sets will be created" 
--published_folder_path "path to patient_based_classified_images" 
--test_set_ratio 0.15 
--val_set_ratio 0.15
```  
1.2 Run `train_classification_model.py` or `train_regression_model.py` indicating train and validation sets.  
Example:
```
python train_classification_model.py 
--train_dir "path to train folder" 
--val_dir "path to validation folder"
```

1.3 After the training finishes, run `inference_classification_based_model.py` or `inference_regression_based_model.py` indicating path to train/test sets and model weights (checkpoints).  
Example:
```
python inference_classification_based_model.py 
--train_dir "path to train folder" 
--test_dir "path to test folder" 
--model_name="ResNet18"  
--checkpoint="weights/best_ResNet18.pth.tar"
```

### 1. If you want to train your model on Cross-Validation setting as described in the LIMUC paper:

1.1 After downloading the dataset, run `generate_10_CV_folds_from_json_files.py`. This script will form the same folds as used in the LIMUC paper.
```
python generate_10_cv_folds_from_json_files.py
--json_folder "path/to/cross_validation_folds_train_val_info"
--train_val_folder "path/to/train_and_validation_sets"
--output_folder "path/to/cross validation folds"
```

1.2 Run `train_classification_model_CV.py` or `train_regression_model_CV.py`.
 

## Important Notes

Since each patient has several images, when performing train-val-test splitting, it is highly
recommended to perform it in patient-level because similar images may go in different sets resulting in high val/test set performance.
Therefore, use the dataset splitting scripts in this repository.

