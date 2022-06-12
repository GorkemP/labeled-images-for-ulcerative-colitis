# Created by Gorkem Polat at 13.12.2021
# contact: polatgorkem@gmail.com

import argparse
import os
import shutil
import json

parser = argparse.ArgumentParser(description="This file generates Cross-validation folds from \n"
                                             ".json files. Use it to make objective evaluations \n"
                                             "and fair comparisons with the results in the dataset.")

parser.add_argument("--json_folder", type=str, required=True, help="Root folder path of the json files")
parser.add_argument("--train_val_folder", type=str, required=True,
                    help="Path of train_and_validation_sets folder as downloaded from the dataset repository")
parser.add_argument("--output_folder", type=str, required=True,
                    help="Root folder path for the cross-validation folds. If this folder exists, it will be removed and created again.")

args = parser.parse_args()

json_root_folder = args.json_folder
train_val_folder = args.train_val_folder
output_folder = args.output_folder

if os.path.isdir(output_folder):
    shutil.rmtree(output_folder)
os.mkdir(output_folder)

json_folders = sorted([fold for fold in os.listdir(json_root_folder) if fold.startswith("fold_")])

for json_folder in json_folders:
    print("Creating ", json_folder)

    current_folder = os.path.join(output_folder, json_folder)
    os.mkdir(current_folder)

    train_json_file = os.path.join(json_root_folder, json_folder, "train.json")
    val_json_file = os.path.join(json_root_folder, json_folder, "val.json")

    train_images = {}
    val_images = {}

    # Copy train files
    with open(train_json_file) as f:
        train_images = json.load(f)

    current_folder_train_folder = os.path.join(current_folder, "train")
    os.mkdir(current_folder_train_folder)
    for key_value_pair in train_images.items():
        current_folder_class = os.path.join(current_folder_train_folder, key_value_pair[0])
        os.mkdir(current_folder_class)

        for file in key_value_pair[1]:
            source_path = os.path.join(train_val_folder, key_value_pair[0], file)
            destination_path = os.path.join(current_folder_train_folder, key_value_pair[0], file)

            shutil.copyfile(source_path, destination_path)

    # Copy validation files
    with open(val_json_file) as f:
        val_images = json.load(f)

    current_folder_val_folder = os.path.join(current_folder, "val")
    os.mkdir(current_folder_val_folder)
    for key_value_pair in val_images.items():
        current_folder_class = os.path.join(current_folder_val_folder, key_value_pair[0])
        os.mkdir(current_folder_class)

        for file in key_value_pair[1]:
            source_path = os.path.join(train_val_folder, key_value_pair[0], file)
            destination_path = os.path.join(current_folder_val_folder, key_value_pair[0], file)

            shutil.copyfile(source_path, destination_path)

print("Folds are created!")
