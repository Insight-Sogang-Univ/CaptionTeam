# Class for Embedding

from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
import torch

# Input으로 token list를 받아서, 학습된 model을 출력하는 class 만들어보기~ class가 불필요하면 함수로 구현해도 될 듯.

# Input으로 token list를 받음!
class embedding_caption():
    def __init__(self,sentences,vector_size=256,window=1,min_count=2,sg=1,method="w2v"):
        self.sentences=sentences
        self.vector_size=vector_size
        self.window=window
        self.min_count=min_count
        self.sg=sg
        self.method=method
        if method=='w2v':
            self.model=Word2Vec(sentences = self.sentences, vector_size = self.vector_size, \
                window = self.window, min_count = self.min_count, sg = self.sg)
        elif method=='fast':
            self.model=FastText(sentences = self.sentences, vector_size = self.vector_size, \
                window = self.window, min_count = self.min_count, sg = self.sg)
    def return_vectors(self):
        return torch.from_numpy(self.model.wv.vectors)
    def save_file(self):
        self.model.save('embedding_model')
