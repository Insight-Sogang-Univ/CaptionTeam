import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import encoder, decoder
from datasets.dataset import CelebDataset
from datasets.preprocess.embedding import CaptionEmbedder
from models.encoder_to_decoder import EncodertoDecoder

if __name__=='__main__':
    
    # Embedder parameter 
    VECTOR_DIM = 256
    DIC_PATH = 'C:\mecab\mecab-ko-dic'
    
    # Embedder 초기화
    ft_embedder = CaptionEmbedder(vector_size=VECTOR_DIM)
    ft_embedder.fit('captions.csv',DIC_PATH)
    VOCAB_SIZE = ft_embedder.vocab_size
    
    # Dataset 정의
    train_data = CelebDataset('datasets/debug/captions.csv','datasets/debug/img',embedder=ft_embedder,dic_path=DIC_PATH)
    
    # Model Parameters
    BATCH_SIZE = 16
    
    ## result through Encoder
    # train_loader에서 (image,caption)을 16장씩 뽑아줌
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True) 
    # next, iter를 통해 loader로부터 image, caption batch 추출
    images, captions = next(iter(train_loader))
    
    
    model = EncodertoDecoder(VECTOR_DIM, VECTOR_DIM, VOCAB_SIZE, num_layers=2)
    result = model(images, torch.zeros((16,20,VECTOR_DIM)))
    
    result2 = model.caption_image(images[0], ft_embedder.vocab.TEXT.vocab, ft_embedder, max_length=20)
    print(len(result2))
    print(result2)