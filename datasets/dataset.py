import os
import pandas as pd
import numpy as np
import torch
from gensim.models import Word2Vec
#from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchtext.legacy.data import Iterator
from datasets.preprocess import vocab

class CelebDataset(Dataset):
    def __init__(self, captions_path:str, img_dir:str, embedder=None, fixed_length=20, dic_path=None):
        '''
        captions_path : dataframe which contains file path - 'head','name' and caption tokens - 'tokenized_captions'
        ''' 
        self.data = pd.read_csv(captions_path)
        self.img_dir = img_dir

        self.captionTokens = vocab.VocabBuilder(dic_path, fixed_length).tokenize_df(captions_path)
        self.fixed_length = fixed_length
        
        self.set_embedder(embedder)
        
        self.transform = transforms.Compose([
            transforms.Resize((299,299))
            ])
                                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir, self.data.loc[idx, 'head'], self.data.loc[idx, 'name'])
        image = read_image(img_path)
        
        indices = self.get_indexed_caption(idx)
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label=self.target_tranform(indices)
        else:
            label=indices
            
        return image, label
    
    def get_raw_caption(self, idx):
        label = self.data['caption'][idx]
        return label
    
    def get_indexed_caption(self, idx):
        label = self.captionTokens[idx]
        label = [self.word2idx.get(word, 0) for word in label]
        
        # Pad the label
        if len(label)<self.fixed_length:
            [label.append(1) for _ in range(self.fixed_length - len(label))]
        else:
            label = label[:self.fixed_length]
        label = torch.tensor(label, dtype=torch.int64)
        
        return label
    
    def set_embedder(self, embedder):
        self.embedder = embedder
        self.vocabs = self.embedder.vocab
        self.word2idx = dict(self.vocabs.TEXT.vocab.stoi)
        self.idx2word=dict([(value, key) for key, value in self.word2idx.items()])
        self.target_transform = self.transform_by_dict
        self.vector_size = embedder.vector_size
        
    def transform_by_dict(self, indices):
        label=torch.zeros((self.fixed_length, self.vector_size))
        for i, w_idx in enumerate(indices.tolist()):
            if self.idx2word[w_idx]=='<unk>':
                label[i] = torch.zeros(self.vector_size)
            elif self.idx2word[w_idx]=='<pad>':
                label[i] = torch.zeros(self.vector_size)
            else:
                label[i] = torch.from_numpy(self.embedder.model.wv[self.idx2word[w_idx]].copy())
        return label