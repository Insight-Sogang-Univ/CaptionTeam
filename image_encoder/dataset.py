import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image

class CelebDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.captions = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.captions.loc[idx, 'head'], self.captions.loc[idx, 'name'])
        #image = read_image(img_path)
        image = Image.open(img_path)
        label = self.captions.loc[idx, 'caption']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label