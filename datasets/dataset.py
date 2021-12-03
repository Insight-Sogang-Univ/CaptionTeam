import os
import pandas as pd
import numpy as np
import torch
from gensim.models import Word2Vec
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchtext.legacy.data import Iterator
from preprocess import vocab

class CelebDataset(Dataset):
    def __init__(self, captions_path:str, img_dir:str, embedder=None, fixed_length=20, dic_path=None, transform=None):
        '''
        captions_path : dataframe which contains file path - 'head','name' and caption tokens - 'tokenized_captions'
        ''' 
        self.data = pd.read_csv(captions_path)
        self.img_dir = img_dir

        self.vocabs = vocab.VocabBuilder(dic_path, fixed_length)
        self.vocabs.tokenize_df(captions_path)
        self.fixed_length = fixed_length
        self.word2idx = dict(self.vocabs.TEXT.vocab.stoi)
        self.target_transform = embedder
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir, self.data.loc[idx, 'head'], self.data.loc[idx, 'name'])
        image = Image.open(img_path)
        
        label = self.vocabs.captionTokens.examples[idx].caption
        label = [self.word2idx.get(word, 0) for word in label]
        # Pad the label
        if len(label)<self.fixed_length:
            [label.append(1) for _ in range(self.fixed_length - len(label))]
        else:
            label = label[:self.fixed_length]
        label = torch.Tensor(label)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def set_embedder(self, embedder):
        self.target_transform = embedder
        
    def idx2word(self):
        return dict([(value, key) for key, value in self.word2idx.items()])