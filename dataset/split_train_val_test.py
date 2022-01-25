import glob
import os
from distutils.dir_util import copy_tree
import numpy as np
from shutil import move
import math
import shutil
import random

np.random.seed(35)
random.seed(35)

root_dir = "/home/ws2080/Desktop/data/ucmayo4_gan"
original_folder = "final_annotations"
test_ratio = 15
val_ratio = 15
# So, train ratio = 100 - (test_ratio + val_ratio)

# Delete if there are folders named train, test, val
if os.path.isdir(os.path.join(root_dir, "train")):
    shutil.rmtree(os.path.join(root_dir, "train"))
if os.path.isdir(os.path.join(root_dir, "test")):
    shutil.rmtree(os.path.join(root_dir, "test"))
if os.path.isdir(os.path.join(root_dir, "val")):
    shutil.rmtree(os.path.join(root_dir, "val"))

subFolders = glob.glob(os.path.join(root_dir, original_folder, "*"))
subFolders.sort()

# At first, train folder is copy of the original folder. Test and validation folders
# will take images from the train folder

# Create train folder
copy_tree(os.path.join(root_dir, original_folder), os.path.join(root_dir, "train"))
# Create val folder
os.mkdir(os.path.join(root_dir, "val"))
# Create test folder
os.mkdir(os.path.join(root_dir, "test"))

for folder in subFolders:
    folder_name = folder.split("/")[-1]
    image_paths = glob.glob(os.path.join(folder, "*"))
    print("Total images in folder " + folder_name + ": " + str(len(image_paths)))

    random_ordering = np.random.permutation(len(image_paths))

    # move from train to test folder
    test_size = math.ceil(len(image_paths) * test_ratio / 100)
    os.mkdir(os.path.join(root_dir, "test", folder_name))

    for i in range(test_size):
        file_name = image_paths[random_ordering[i]].split("/")[-1]
        move(os.path.join(root_dir, "train", folder_name, file_name), os.path.join(root_dir, "test", folder_name,
                                                                                   file_name))
    print("----> Test Set: For folder " + folder_name + ", " + str(test_size) + " files are copied")

    # move from train to val folder
    val_size = math.ceil(len(image_paths) * val_ratio / 100)
    os.mkdir(os.path.join(root_dir, "val", folder_name))

    for i in range(test_size, test_size + val_size):
        file_name = image_paths[random_ordering[i]].split("/")[-1]
        move(os.path.join(root_dir, "train", folder_name, file_name), os.path.join(root_dir, "val", folder_name,
                                                                                   file_name))
    print("----> Validation Set: For folder " + folder_name + ", " + str(val_size) + " files are copied")
