import torch
from torch.utils.data import DataLoader

from models import encoder, decoder
from datasets.dataset import CelebDataset
from datasets.preprocess.embedding import CaptionEmbedder

if __name__=='__main__':
    # Embedder parameter
    VECTOR_DIM = 256
    DIC_PATH = 'C:\mecab\mecab-ko-dic'
    
    # Embedder 초기화
    ft_embedder = CaptionEmbedder(vector_size=VECTOR_DIM)
    ft_embedder.fit('captions.csv',DIC_PATH)
    
    # Dataset 정의
    train_data = CelebDataset('datasets/debug/captions.csv','datasets/debug/img',embedder=ft_embedder,dic_path=DIC_PATH)
    
    # Model Parameters
    BATCH_SIZE = 16
    
    ## result through Encoder
    # train_loader에서 (image,caption)을 16장씩 뽑아줌
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True) 
    # next, iter를 통해 loader로부터 image, caption batch 추출
    images, captions = next(iter(train_loader))
    # VECROT_DIM size의 output을 반환하는 Encoder 정의
    Encoder = encoder.EncoderInception3(embed_size = VECTOR_DIM)
    # image를 Encoder에 forwarding한 결과 features 반환
    features = Encoder(images)
    
    print('train loader의 images{}가 Encoder를 통과한 결과 features{}'.format(images.size(),features.size()))
    print(features)