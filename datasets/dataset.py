# import os
# import numpy as np
import torch
import h5py
#from PIL import Image
from torch.utils.data import Dataset
# from torchvision import transforms
# from torchvision.io import read_image

class CelebDataset(Dataset):
    def __init__(self, df, img_file:str, embedder, fixed_length=20, transform=None):
        '''
        captions_path : dataframe which contains file path - 'head','name' and caption tokens - 'tokenized_captions'
        ''' 
        self.data = df
        self.img_file = h5py.File(img_file)

        self.embedder = embedder
        self.w2i = self.embedder.w2i
        self.i2w = self.embedder.i2w
        
        self.fixed_length = fixed_length
        self.transform = transform
                                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image = self.img_file.get(self.data.loc[idx, 'head']).get(self.data.loc[idx, 'name'])
        image = torch.from_numpy(image[:]).type(torch.FloatTensor)
    
        if self.transform:
            image = self.transform(image)
        
        # Index
        label = list(map(int,self.data.loc[idx, 'indexed'].split()))
        # PAD
        if len(label) < self.fixed_length:
            [label.append(self.w2i['<pad>']) for _ in range(self.fixed_length - len(label))]
        label = label[:self.fixed_length]
        label = torch.tensor(label)
        
        # Vectorize
        vectors = torch.zeros((self.fixed_length, self.embedder.vector_size))
        for i, idx in enumerate(label):
            vectors[i] = self.vectorize_caption(idx.item())
        vectors = vectors.type(torch.FloatTensor)
        
        return image, label, vectors
    
    def get_raw_caption(self, idx):
        label = self.data.loc[idx,'caption']
        return label
    
    def vectorize_caption(self, idx):
        return self.embedder.vectorize_caption(idx)