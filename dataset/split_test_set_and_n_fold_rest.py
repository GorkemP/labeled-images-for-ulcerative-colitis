# Created by Gorkem Polat at 1.06.2021
# contact: polatgorkem@gmail.com

import os
import glob
import random
from random import shuffle
import shutil
from distutils.dir_util import copy_tree
from math import ceil
import numpy as np
import argparse

# fix seed to get same folds for reproducibility
random.seed(35)
np.random.seed(35)

parser = argparse.ArgumentParser(description="This script generates n-folds for cross-validation. \n"
                                             "First, it forms a test set by given test_set_ratio. \n"
                                             "Then, it creates n train-val folds. \n"
                                             "validation and test sets ratios may not be exactly as \n"
                                             "it is given in ratios due to patient-level splitting\n")

parser.add_argument("--CV_folder_path", type=str, required=True,
                    help="CV folds and test set folder will be created in this path.")
parser.add_argument("--published_folder_path", type=str, required=True,
                    help="path of patient_based_classified_images folder.")
parser.add_argument("--fold_num", type=int, default=10, help="Number of folds.")
parser.add_argument("--test_set_ratio", type=float, default=0.2, help="ratio of the test set.")
parser.add_argument("--test_set_error_ratio", type=float, default=0.15, help="tolerance of the test set ratio.")
parser.add_argument("--error_ratio", type=float, default=0.15,
                    help="tolerance level for validation set ratios.")

args = parser.parse_args()

CV_folder_path = args.CV_folder_path
published_folder_path = args.published_folder_path

fold_num = args.fold_num
fold_folder_prefix = "fold"
all_folders = os.listdir(published_folder_path)
folder_class_names = sorted(os.listdir(os.path.join(published_folder_path, all_folders[0])))

error_ratio = args.error_ratio
test_set_error_ratio = args.test_set_error_ratio
test_set_ratio = args.test_set_ratio
val_set_ratio = 1. / fold_num

if os.path.isdir(CV_folder_path):
    shutil.rmtree(CV_folder_path)
os.mkdir(CV_folder_path)

# Get size for all classes
class_sizes = {}
for class_name in folder_class_names:
    class_sizes[class_name] = 0

for folder in all_folders:
    for class_name in folder_class_names:
        class_sizes[class_name] += len(glob.glob(os.path.join(published_folder_path, folder, class_name, "*.bmp")))

# Just print class sizes to check
for item in class_sizes:
    print(str(item) + ": " + str(class_sizes[item]))

# first form the test set with proper class ratios and set it aside
test_set_pass = False
train_val_folders = []
while not test_set_pass:
    test_set_pass = True

    target_test_path = os.path.join(CV_folder_path, "test")
    if os.path.isdir(target_test_path):
        shutil.rmtree(target_test_path)
    os.mkdir(target_test_path)

    shuffle(all_folders)

    test_set_folder_size = ceil(test_set_ratio * len(all_folders))
    test_folders = all_folders[:test_set_folder_size]
    train_val_folders = all_folders[test_set_folder_size:]

    for class_name in folder_class_names:
        os.mkdir(os.path.join(target_test_path, class_name))

    for test_folder in test_folders:
        for class_folder in os.scandir(os.path.join(published_folder_path, test_folder)):
            target_directory = os.path.join(target_test_path, class_folder.name)
            copy_tree(class_folder.path, target_directory)

    print("\nChecking test set ratios:")
    for class_name in folder_class_names:
        test_size = len(glob.glob(os.path.join(target_test_path, class_name, "*.bmp")))
        total = class_sizes[class_name]

        if ((test_size / total) < test_set_ratio * (1 - test_set_error_ratio)) or (
                (test_size / total) > test_set_ratio * (1 + test_set_error_ratio)):
            print("unbalanced distribution in test set for class " + class_name + ". Recalculating!")
            print("test size: {:3.1f}".format((test_size / total) * 100))
            test_set_pass = False
            break

    if test_set_pass:
        print("Successful distribution in test set.")
        for class_name in folder_class_names:
            test_size = len(glob.glob(os.path.join(target_test_path, class_name, "*.bmp")))
            print(class_name + ": " + str(test_size) + " | test size: {:3.1f}".format((test_size / class_sizes[
                class_name]) * 100))

i = 0
while i < fold_num:
    pass_state = True

    target_fold_path = os.path.join(CV_folder_path, fold_folder_prefix + "_" + str(i))
    if os.path.isdir(target_fold_path):
        shutil.rmtree(target_fold_path)
    os.mkdir(target_fold_path)

    shuffle(train_val_folders)

    val_set_folder_size = ceil(val_set_ratio * len(train_val_folders))
    val_folders = train_val_folders[0:val_set_folder_size]
    train_folders = train_val_folders[val_set_folder_size:]

    # Create train folder
    target_train_path = os.path.join(target_fold_path, "train")
    os.mkdir(target_train_path)
    for class_name in folder_class_names:
        os.mkdir(os.path.join(target_train_path, class_name))

    for train_folder in train_folders:
        for class_folder in os.scandir(os.path.join(published_folder_path, train_folder)):
            target_directory = os.path.join(target_train_path, class_folder.name)
            copy_tree(class_folder.path, target_directory)

    # Create val folder
    target_val_path = os.path.join(target_fold_path, "val")
    os.mkdir(target_val_path)
    for class_name in folder_class_names:
        os.mkdir(os.path.join(target_val_path, class_name))

    for val_folder in val_folders:
        for class_folder in os.scandir(os.path.join(published_folder_path, val_folder)):
            target_directory = os.path.join(target_val_path, class_folder.name)
            copy_tree(class_folder.path, target_directory)

    print("\nChecking splitting ratios for folds:")
    for class_name in folder_class_names:
        train_size = len(glob.glob(os.path.join(target_train_path, class_name, "*.bmp")))
        val_size = len(glob.glob(os.path.join(target_val_path, class_name, "*.bmp")))
        test_size = len(glob.glob(os.path.join(CV_folder_path, "test", class_name, "*.bmp")))
        total = class_sizes[class_name]

        if ((val_size / total) < (val_set_ratio * (1 - test_size / total) * (1 - error_ratio))) or (
                (val_size / total) > (val_set_ratio * (1 - test_size / total) * (1 + error_ratio))):
            print("Unbalanced distribution in fold " + str(i) + " for class " + class_name + ". Recalculating!")
            print("val size: {:3.1f}".format((val_size / total) * 100))
            pass_state = False
            break

    if pass_state:
        print("Successful distribution in fold " + str(i) + ". Advancing to next fold.")
        i += 1

print("\nClass ratios for val set:")
for fold in os.scandir(CV_folder_path):
    if fold.name.startswith("fold"):
        print("\n" + fold.name)
        target_train_path = os.path.join(fold.path, "train")
        target_val_path = os.path.join(fold.path, "val")

        total_classes = 0
        for class_name in folder_class_names:
            train_size = len(glob.glob(os.path.join(target_train_path, class_name, "*.bmp")))
            val_size = len(glob.glob(os.path.join(target_val_path, class_name, "*.bmp")))
            total = train_size + val_size
            total_classes += total

            print(
                    class_name + ": " + " val set ratio: {:3.1f}".format(
                            (val_size / total) * 100) + " => total: " + str(
                            total))
        print("All images: " + str(total_classes))
