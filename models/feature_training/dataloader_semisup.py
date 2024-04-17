import os
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import cv2
syn_mean = np.array([0.8069703419651933, 0.8611282975935172, 0.912551699958795])
syn_std = np.array([0.15258447275960968, 0.13846048412015757, 0.09557870057351356])


transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size = (320, 320))
    #transforms.Normalize(syn_mean, syn_std)
])

class SynDatasetUns(data.Dataset):

    """

    A simple dataloader for Synthetic Images. Requires a datapath for original images and labels.

    """

    def __init__(self,
                 data_path, label_path):
        self.data_path = data_path
        self.label_path = label_path
        self.filenames = os.listdir(data_path)
        self.labels = os.listdir(label_path)
        self.filenames.sort()
        self.labels.sort()

    def __getitem__(self, index):
        with open(os.path.join(self.data_path,self.filenames[index]), 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        input_image = np.array(img)

        input_image = transform_norm(input_image)
        label = cv2.imread(self.label_path + self.labels[index])

        return {
            "image" : input_image,
            "label" : label
        }

    def __len__(self):
        return len(self.filenames)
