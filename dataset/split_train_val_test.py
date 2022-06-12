import glob
import os
from distutils.dir_util import copy_tree
import numpy as np
from math import ceil
from random import shuffle
import shutil
import random
import argparse

np.random.seed(35)
random.seed(35)

parser = argparse.ArgumentParser(description="This script generates train-val-test sets by splitting \n"
                                             "original dataset (patient_based_classified_images) in \n"
                                             "patient-level. Due to patient-level splitting, train-val-test \n"
                                             "sets ratios may not be exactly as it is given in ratios.\n")

parser.add_argument("--target_dir", type=str, required=True,
                    help="train, val, test folders will be created in this folder.")
parser.add_argument("--published_folder_path", type=str, required=True,
                    help="path of patient_based_classified_images folder.")
parser.add_argument("--test_set_ratio", type=float, default=0.2, help="ratio of the test set.")
parser.add_argument("--val_set_ratio", type=float, default=0.2, help="ratio of the val set.")
parser.add_argument("--error_ratio", type=float, default=0.15,
                    help="tolerance level for test and validation set ratios.")

args = parser.parse_args()

target_dir = args.target_dir
published_folder_path = args.published_folder_path
test_set_ratio = float(args.test_set_ratio)
val_set_ratio = float(args.val_set_ratio)
error_ratio = float(args.error_ratio)

all_folders = os.listdir(published_folder_path)
folder_class_names = sorted(os.listdir(os.path.join(published_folder_path, all_folders[0])))

pass_state = False

# Remove all folders inside patient Mayo folders, there should be only images
for folder in all_folders:
    for folder_class_name in folder_class_names:
        class_path = os.path.join(published_folder_path, folder, folder_class_name)
        for item in os.scandir(class_path):
            if item.is_dir():
                print("removing redundant folder:", item.path)
                shutil.rmtree(item.path)

while not pass_state:
    pass_state = True

    # Delete if there are folders named train, test, val in target directory
    if os.path.isdir(os.path.join(target_dir, "train")):
        shutil.rmtree(os.path.join(target_dir, "train"))
    if os.path.isdir(os.path.join(target_dir, "test")):
        shutil.rmtree(os.path.join(target_dir, "test"))
    if os.path.isdir(os.path.join(target_dir, "val")):
        shutil.rmtree(os.path.join(target_dir, "val"))

    shuffle(all_folders)

    val_set_folder_size = ceil(val_set_ratio * len(all_folders))
    test_set_folder_size = ceil(test_set_ratio * len(all_folders))

    val_folders = all_folders[0:val_set_folder_size]
    test_folders = all_folders[val_set_folder_size:(val_set_folder_size + test_set_folder_size)]
    train_folders = all_folders[(val_set_folder_size + test_set_folder_size):]

    # Create train folder
    target_train_path = os.path.join(target_dir, "train")
    os.mkdir(target_train_path)
    for class_name in folder_class_names:
        os.mkdir(os.path.join(target_train_path, class_name))

    for train_folder in train_folders:
        for class_folder in os.scandir(os.path.join(published_folder_path, train_folder)):
            target_directory = os.path.join(target_train_path, class_folder.name)
            copy_tree(class_folder.path, target_directory)

    # Create val folder
    target_val_path = os.path.join(target_dir, "val")
    os.mkdir(target_val_path)
    for class_name in folder_class_names:
        os.mkdir(os.path.join(target_val_path, class_name))

    for val_folder in val_folders:
        for class_folder in os.scandir(os.path.join(published_folder_path, val_folder)):
            target_directory = os.path.join(target_val_path, class_folder.name)
            copy_tree(class_folder.path, target_directory)

    # Create test folder
    target_test_path = os.path.join(target_dir, "test")
    os.mkdir(target_test_path)
    for class_name in folder_class_names:
        os.mkdir(os.path.join(target_test_path, class_name))

    for test_folder in test_folders:
        for class_folder in os.scandir(os.path.join(published_folder_path, test_folder)):
            target_directory = os.path.join(target_test_path, class_folder.name)
            copy_tree(class_folder.path, target_directory)

    print("\nChecking splitting ratios:")
    for class_name in folder_class_names:
        print(class_name)
        train_size = len(glob.glob(os.path.join(target_train_path, class_name, "*.bmp")))
        val_size = len(glob.glob(os.path.join(target_val_path, class_name, "*.bmp")))
        test_size = len(glob.glob(os.path.join(target_test_path, class_name, "*.bmp")))
        total = train_size + val_size + test_size

        if ((test_size / total) < test_set_ratio * (1 - error_ratio)) or (
                (test_size / total) > test_set_ratio * (1 + error_ratio)) \
                or ((val_size / total) < val_set_ratio * (1 - error_ratio)) or (
                (val_size / total) > val_set_ratio * (1 + error_ratio)):
            print("test size: {:4.3f}".format(test_size / total))
            print("val size: {:4.3f}".format(val_size / total))
            print("Unbalanced distribution for class " + class_name + ". Recalculating!")
            pass_state = False
            break

    if pass_state:
        print("Successful distribution!")

print("\nClass ratios for test and val set:")
target_train_path = os.path.join(target_dir, "train")
target_val_path = os.path.join(target_dir, "val")
target_test_path = os.path.join(target_dir, "test")

total_classes = 0
for class_name in folder_class_names:
    train_size = len(glob.glob(os.path.join(target_train_path, class_name, "*.bmp")))
    val_size = len(glob.glob(os.path.join(target_val_path, class_name, "*.bmp")))
    test_size = len(glob.glob(os.path.join(target_test_path, class_name, "*.bmp")))
    total = train_size + val_size + test_size
    total_classes += total

    print(class_name + ": " + " test set ratio: {:4.3f}".format(test_size / total) + " val set ratio: {:4.3f}".format(
            val_size / total) + " => total: " + str(total))
print("All images: " + str(total_classes))
