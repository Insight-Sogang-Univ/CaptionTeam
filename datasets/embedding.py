# Class for Embedding
import numpy as np
import pandas as pd

import torch
import pickle
import sys
from config.train_config import DIC_PATH
sys.path.insert(0,'..')

from preprocess.vocab import VocabBuilder2

from gensim.models.word2vec import Word2Vec
from gensim.models import FastText

class CaptionEmbedder():
    def __init__(self, vector_size=256, window=3, min_count=3, sg=1):
        self.model = None
        self.w2i = None
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        
    @property
    def i2w(self):
        return dict([(value, key) for key, value in self.w2i.items()])
        
    def fit(self, df, method='fast'):
        
        df, self.w2i = self.process_df(df)
        
        captions = df['tokenized'].apply(lambda x: x.split())
        
        if method=='w2v':
            self.model=Word2Vec(sentences = captions, vector_size = self.vector_size, \
                window = self.window, min_count = self.min_count, sg = self.sg)
        elif method=='fast':
            self.model=FastText(sentences = captions, vector_size = self.vector_size, \
                window = self.window, min_count = self.min_count, sg = self.sg)
            
        return df
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
        
    def process_df(self, df):
        df = df.copy()
        builder = VocabBuilder2(DIC_PATH)
        df = builder.tokenize_df(df)
        w2i = VocabBuilder2.make_dict(df['tokenized'], self.min_count)
        df = builder.indexize_df(df, w2i)
        return df, w2i
        
    def vectorize_caption(self, idx):
        if type(idx)==torch.Tensor:
            idx = idx.item()
        if self.i2w[idx]=='<unk>':
            return torch.zeros(self.vector_size)
        elif self.i2w[idx]=='<pad>':
            return torch.zeros(self.vector_size)
        else:
            return torch.from_numpy(self.model.wv[self.i2w[idx]].copy())