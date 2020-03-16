import os
import numpy as np

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset


class PACS(Dataset):
    
    def __init__(self, root, split_dir, domain, split, transform=None):
        self.root = os.path.expanduser(root)
        self.split_dir = os.path.expanduser(split_dir)
        self.domain = domain
        self.split = split
        self.transform = transform
        self.loader = default_loader

        self.preprocess()

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.samples[index][0])
        label = self.samples[index][1]
        
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.samples)

    def preprocess(self):
        split_path = os.path.join(self.split_dir, '{}_{}_kfold.txt'.format(self.domain, self.split))
        self.samples = np.genfromtxt(split_path, dtype=str).tolist()
        self.samples = [(img, int(lbl) - 1) for img, lbl in self.samples]
        
        print('domain: {:14s} split: {:10s} n_images: {:<6d}'
                .format(self.domain, self.split, len(self.samples)))
