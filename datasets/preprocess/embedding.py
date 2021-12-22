# Class for Embedding
import torch
import pandas as pd

import os
import pickle
from config import *

from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
from datasets.preprocess.vocab import VocabBuilder

class CaptionEmbedder():
    def __init__(self):
        self.vocab = None
        self.model = None
        self.vector_size = VECTOR_DIM
        
    def fit(self):
        method = METHOD
        vector_size = VECTOR_DIM
        window = WINDOW_SIZE
        min_count = MIN_COUNT
        sg = SG
        
        self.vocab = VocabBuilder(DIC_PATH)
        self.vocab.tokenize_df(LABEL_PATH)
        
        if method=='w2v':
            self.model=Word2Vec(sentences = self.captions, vector_size = vector_size, \
                window = window, min_count = min_count, sg = sg)
        elif method=='fast':
            self.model=FastText(sentences = self.captions, vector_size = vector_size, \
                window = window, min_count = min_count, sg = sg)
    
    @property
    def captions(self):
        return self.vocab.captionTokens
    
    @property
    def vocab_size(self):
        return len(self.vocab.TEXT.vocab.stoi)
        
    @property
    def word2idx(self):
        return dict(self.vocab.TEXT.vocab.stoi)
    
    @property
    def idx2word(self):
        return dict([(value, key) for key, value in dict(self.vocab.TEXT.vocab.stoi).items()])
            
    def return_vectors(self):
        return torch.from_numpy(self.model.wv.vectors)
    
    def save(self):

        if os.path.exists(EMBED_PATH):
            pass
        else:
            os.mkdir(EMBED_PATH)

        with open(EMBED_PATH + 'embed_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open(EMBED_PATH + 'embed_vocab.pkl', 'wb') as f:
            pickle.dump(self.vocab, f)

    def load(self):
        with open(EMBED_PATH + 'embed_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open(EMBED_PATH + 'embed_vocab.pkl', 'rb') as f:
            self.vocab = pickle.load(f)
        
    def transform_by_dict(self, idx):
        if self.idx2word[idx]=='<unk>':
            return torch.zeros(self.vector_size)
        elif self.idx2word[idx]=='<pad>':
            return torch.zeros(self.vector_size)
        else:
            return torch.from_numpy(self.model.wv[self.idx2word[idx]].copy())