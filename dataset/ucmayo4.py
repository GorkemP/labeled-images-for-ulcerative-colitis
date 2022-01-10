import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob

class UCMayo4(Dataset):
    """Ulcerative Colitis dataset grouped according to Endoscopic Mayo scoring system"""

    def __init__(self, root_dir, transform=None):
        """

        root_dir (string): Path to parent folder where class folders are located.
        transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.class_names = []
        self.samples = []
        self.transform = transform

        subFolders = glob.glob(os.path.join(root_dir, "*"))
        subFolders.sort()

        for folder in subFolders:
            className = folder.split("/")[-1]
            self.class_names.append(className)

        self.number_of_class = len(self.class_names)

        for folder in subFolders:
            className = folder.split("/")[-1]
            image_paths = glob.glob(os.path.join(folder, "*"))

            for image_path in image_paths:
                image = Image.open(image_path)
                image.load()
                self.samples.append((image, self.class_names.index(className)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_image = self.samples[idx][0].copy()
        if self.transform:
            sample_image = self.transform(sample_image)

        return (sample_image, self.samples[idx][1])

class UCMayo4Remission(Dataset):
    """
    Ulcerative Colitis dataset grouped according to Endoscopic Mayo scoring system
    According to the remission list given in constructor, it has binary output for annotation.
    """

    def __init__(self, root_dir, remission=[2, 3], transform=None):
        """
        Args:
            root_dir (string): Path to parent folder where class folders are located.
            resmission (list): Mayo scores (as int) that will be regarded as non-remission state.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.number_of_class = 2
        self.class_names = []
        self.samples = []
        self.transform = transform

        subFolders = glob.glob(os.path.join(root_dir, "*"))
        subFolders.sort()

        for folder in subFolders:
            className = folder.split("/")[-1]
            self.class_names.append(className)

        for folder in subFolders:
            className = folder.split("/")[-1]
            image_paths = glob.glob(os.path.join(folder, "*"))

            for image_path in image_paths:
                image = Image.open(image_path)
                image.load()

                label = 0
                if self.class_names.index(className) in remission:
                    label = 1
                self.samples.append((image, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_image = self.samples[idx][0].copy()
        # TODO since all images are loaded at constructor, transform can be moved there too
        if self.transform:
            sample_image = self.transform(sample_image)

        return (sample_image, self.samples[idx][1])
