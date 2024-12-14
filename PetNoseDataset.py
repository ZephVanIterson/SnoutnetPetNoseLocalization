import os
import ast
import pandas as pd
import torch
from torchvision.io import read_image

class PetNoseDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        image = read_image(img_path)

        self.width = image.shape[1]
        self.height = image.shape[2]
        # self.img_name = img_path.split('/')[-1]

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        # Convert label to list and scale label to range 0-227
        label = list(ast.literal_eval(label))

        label[0] = round(label[0] * 227 / self.height)
        label[1] = round(label[1] * 227 / self.width)

        if self.target_transform == "Horizontal Flip":
            label = list(label)
            label[0] = 227 - label[0]

        # Convert label to tensor
        label = torch.tensor(label).float()
        image = image.float()
        return (image, label)
