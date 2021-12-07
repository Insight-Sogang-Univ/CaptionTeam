# Class for Embedding

import torch
import pandas as pd
from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
from datasets.preprocess.vocab import VocabBuilder

class CaptionEmbedder():
    def __init__(self,vector_size=256,window=1,min_count=2,sg=1,method="w2v"):
        self.captions=None
        self.vector_size=vector_size
        self.window=window
        self.min_count=min_count
        self.sg=sg
        self.method=method
        self.vocab=None
        self.model=None
        
    def fit(self, caption_path, tokenizer_dic_path):
        #Tokenize
        self.vocab = VocabBuilder(tokenizer_dic_path)
        self.vocab.tokenize_df(caption_path)
        self.vocab_size = len(self.vocab.TEXT.vocab.stoi)
        self.captions = self.vocab.captionTokens
        
        if self.method=='w2v':
            self.model=Word2Vec(sentences = self.captions, vector_size = self.vector_size, \
                window = self.window, min_count = self.min_count, sg = self.sg)
        elif self.method=='fast':
            self.model=FastText(sentences = self.captions, vector_size = self.vector_size, \
                window = self.window, min_count = self.min_count, sg = self.sg)
        self.word2idx = dict(self.vocab.TEXT.vocab.stoi)
        self.idx2word=dict([(value, key) for key, value in self.word2idx.items()])
            
    def return_vectors(self):
        return torch.from_numpy(self.model.wv.vectors)
    
    def save(self, file_name):
        self.model.save(file_name)
        
    def transform_by_dict(self, idx):
        if self.idx2word[idx]=='<unk>':
            return torch.zeros(self.vector_size)
        elif self.idx2word[idx]=='<pad>':
            return torch.zeros(self.vector_size)
        else:
            return torch.from_numpy(self.model.wv[self.idx2word[idx]].copy())