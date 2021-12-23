import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

import os
import time
import pickle
import argparse
from config import *
from tqdm import tqdm

from models import encoder, decoder
from datasets.dataset import CelebDataset
from datasets.preprocess.embedding import CaptionEmbedder
from models.encoder_to_decoder import EncodertoDecoder

def train(model, dataloader, criterion, optimizer, epoch=0):
    log_format = "epoch: {:4d}, step: {:4d}/{:4d}, loss: {:.6f}, " \
                              "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m, lr: {:.6f}"
    cers = []
    epoch_loss_total = 0.
    total_num = 0
    timestep = 0
    
    begin_time = epoch_begin_time = time.time() #모델 학습 시작 시간

    progress_bar = tqdm(enumerate(dataloader),ncols=110)
    for i, (images, captions, vectors) in progress_bar:
        
        indices = []
        for i in range(images.shape[0]):
          if not torch.equal(images[i],torch.zeros((3,299,299))):
            indices.append(i)
        images = images[indices]
        captions = captions[indices]
        vectors = vectors[indices]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        optimizer.zero_grad()

        images = images.to(device)
        captions = captions.to(device)
        vectors = vectors.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(images, vectors[:,:-1,:])
        
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions.reshape(-1))
        
        loss.backward()
        optimizer.step()

        total_num += len(images)
        epoch_loss_total += loss.item()

        timestep += 1
        torch.cuda.empty_cache()
        
        if timestep % 1 == 0:
            current_time = time.time()
            elapsed = current_time - begin_time
            epoch_elapsed = (current_time - epoch_begin_time) / 60.0
            
            progress_bar.set_description(
                log_format.format(epoch+1,
                timestep, len(dataloader), loss,
                0, elapsed, epoch_elapsed,
                optimizer.state_dict()['param_groups'][0]['lr'])
            )
            begin_time = time.time()
    
    train_loss = epoch_loss_total / total_num
    return train_loss

def get_args():
    parser = argparse.ArgumentParser(description='각종 옵션')
    parser.add_argument('-lr','--learning_rate', required=True,
                        type=float, help='모델이름 입력')
    parser.add_argument('-e','--epochs', required=True,
                        type=int, help='학습횟수 입력')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Embedder 불러오기
    if os.path.exists(EMBED_PATH):
        with open(EMBED_PATH, 'rb+') as f:
            ft_embedder = pickle.load(f)
        print('Embedder Loaded')
    else:
        print('Embedding Start')
        ft_embedder = CaptionEmbedder()
        ft_embedder.fit()
        # ft_embedder.save() # ERROR 잡아야 됨..
        print('Embedding Complete')
    VOCAB_SIZE = ft_embedder.vocab_size
    
    transform = transforms.Compose(
        [
            transforms.Resize((299,299)),
            # transforms.ToTensor(),
            transforms.Normalize((127.5, 127.5, 127.5), (255., 255., 255.))
        ]
    )
    
    print('DataLoading Start')
    train_data = CelebDataset(LABEL_PATH, IMAGE_FILE, embedder=ft_embedder, dic_path=DIC_PATH, transform = transform)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True) 
    print('DataLoading Complete')
    
    model = EncodertoDecoder(VECTOR_DIM, VECTOR_DIM, VOCAB_SIZE, num_layers=2).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index = train_data.word2idx['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    print(model)
    
    train_loss = []
    
    print('Train Start')
    for epoch in range(args.epochs):
        ###################     Train    ###################
        model.train()
        epoch_loss = train(model, train_loader, criterion, optimizer, epoch)
        train_loss.append(epoch_loss)
        print('{}th Train Loss : {}'.format(epoch+1, epoch_loss))
        
        if not SAVE_PATH in os.listdir(os.getcwd()):
            os.mkdir(SAVE_PATH)
        
        torch.save(model, os.path.join(SAVE_PATH, 'checkpoint_epoch_'+str(epoch+1)+'.pt'))
        
        ###################     Valid    ###################
        # model.eval()
        # with torch.no_grad():
        #     pass
        # if scheduler: scheduler.step(val_loss)