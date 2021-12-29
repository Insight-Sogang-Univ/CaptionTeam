import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import os, time, pickle, argparse
import pdb
import pandas as pd
from config.train_config import *
from tqdm import tqdm

from datasets.dataset import CelebDataset
from datasets.embedding import CaptionEmbedder
from models.encoder_to_decoder import EncodertoDecoder

def train(model, dataloader, criterion, optimizer, epoch=0):
    log_format = "epoch: {:4d}, step: {:4d}/{:4d}, loss: {:.6f}, " \
                              "elapsed: {:.2f}s {:.2f}m, lr: {:.6f}"

    epoch_loss_total = 0.
    total_num = 0
    timestep = 0
    
    begin_time = epoch_begin_time = time.time() #모델 학습 시작 시간

    progress_bar = tqdm(enumerate(dataloader),ncols=110)
    for i, (images, captions, vectors) in progress_bar:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        optimizer.zero_grad()

        images = images.to(device)
        captions = captions.to(device)
        vectors = vectors.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(images, vectors[:,:-1,:])
        # pdb.set_trace()
        # loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions.reshape(-1))
        outputs = outputs.argmax(-1)
        print(outputs.unsqueeze(0).size(),
              captions.reshape(1,1,-1).size())
        loss = criterion(outputs.unsqueeze(0).size(),
              captions.reshape(1,1,-1).size())
        
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
                elapsed, epoch_elapsed,
                optimizer.state_dict()['param_groups'][0]['lr'])
            )
            begin_time = time.time()
    
    train_loss = epoch_loss_total / len(dataloader)
    return train_loss

def validate(model, dataloader, criterion):
    epoch_loss_total = 0.
    total_num = 0
    progress_bar = tqdm(enumerate(dataloader),ncols=110)
    for j, (images, captions, vectors) in progress_bar:
        
        indices = []
        for j in range(images.shape[0]):
            if not torch.equal(images[j],torch.zeros((3,299,299))):
                indices.append(j)
        images = images[indices]
        captions = captions[indices]
        vectors = vectors[indices]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        images = images.to(device)
        captions = captions.to(device)
        vectors = vectors.to(device)
    
        with torch.cuda.amp.autocast():
            outputs = model(images, vectors[:,:-1,:])
        
        # loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions.reshape(-1))
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]).unsqueeze(0), 
                         captions.reshape(-1).unsqueeze(0).unsqueeze(0))
        
        total_num += len(images)
        epoch_loss_total += loss.item()
        
        torch.cuda.empty_cache()
    
    valid_loss = epoch_loss_total / len(dataloader)
    return valid_loss


def get_args():
    parser = argparse.ArgumentParser(description='각종 옵션')
    parser.add_argument('-lr','--learning_rate', required=True,
                        type=float, help='학습률 입력')
    parser.add_argument('-e','--epochs', required=True,
                        type=int, help='학습횟수 입력')
    parser.add_argument('-m','--model', default='lstm',
                        type=str, help='사용 모델명 입력')
    parser.add_argument('-re','--resume_from',
                        help='지난 학습 모델 불러오기')
    parser.add_argument('-use','--use',
                        help='학습데이터 지정하기')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()
    
    if args.use:
        use = args.use
        LABEL_PATH = LABEL_PATH[:-4]+f'_{use}.csv'
        EMBED_LABEL = EMBED_LABEL[:-4]+f'_{use}.csv'
        IMAGE_FILE = IMAGE_FILE[:-5]+f'_{use}.hdf5'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Embedder & data 불러오기
    if os.path.exists(EMBED_PATH):
        with open(EMBED_PATH, 'rb+') as f:
            ft_embedder = pickle.load(f)
        print('Embedder Loaded')
    else:
        ft_embedder = CaptionEmbedder(VECTOR_DIM, WINDOW_SIZE, MIN_COUNT, SG)
        df = pd.read_csv(LABEL_PATH)
        df = ft_embedder.fit(df, 'fast')
        ft_embedder.save(EMBED_PATH)
        print('Embedder Fit')
    # Embed Label 불러오기
    if os.path.exists(EMBED_LABEL):
        df = pd.read_csv(EMBED_LABEL)
    else:
        print('Embedding Start')
        df = pd.read_csv(LABEL_PATH)
        df, _ = ft_embedder.process_df(df)
        df.to_csv(EMBED_LABEL, index=False)
        print('Embedding Complete')
    VOCAB_SIZE = len(ft_embedder.w2i)
    
    transform = transforms.Compose(
        [
            transforms.Resize((299,299)),
            # transforms.ToTensor(),
            transforms.Normalize((127.5, 127.5, 127.5), (255., 255., 255.))
        ]
    )
    
    print('DataLoading Start')
    dataset = CelebDataset(df, IMAGE_FILE, embedder=ft_embedder, fixed_length=FIXED_LENGTH, transform = transform)
    
    train_length=int((1-VALID_RATE)*len(dataset))
    valid_length=len(dataset)-train_length
    train_data, valid_data = random_split(dataset, [train_length,valid_length])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    print('DataLoading Complete')
    
    model = EncodertoDecoder(VECTOR_DIM, VECTOR_DIM, VOCAB_SIZE, num_layers=2, model=args.model, embedder=ft_embedder).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.w2i['홍길동']) #(ignore_index = dataset.w2i['<pad>'])
    # criterion = lambda candidate, ref: 1 - bleu_score(candidate, ref)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    print(model)
    
    if args.resume_from:
        # 저장했던 중간 모델 정보를 읽습니다.
        model_data = torch.load(args.resume_from)
        model.load_state_dict(model_data['model_state_dict'])
        
        # optimizer도 중간에 저장했던 값들로 치환합니다.
        optimizer.load_state_dict(model_data['optimizer_state_dict'])
        
        # 지금 시작할 epoch은 기존 epoch + 1 즉 다음 epoch입니다.
        start_epoch = model_data['epoch']+1
        
    else:
        start_epoch = 0
    
    train_loss = []
    valid_loss = []
    print(f'Train Start from {start_epoch+1}th epoch...')
    for epoch in range(start_epoch, args.epochs):
        ###################     Train    ###################
        model.train()
        train_epoch_loss = train(model, train_loader, criterion, optimizer, epoch)
        train_loss.append(train_epoch_loss)
        
        ###################     Valid    ###################
        model.eval()
        valid_epoch_loss = validate(model, valid_loader, criterion)
        valid_loss.append(valid_epoch_loss)
        
        ###################   EPOCH END   #################
        home = os.getcwd()
        for path in SAVE_PATH.split('/'):
            if not os.path.exists(path):
                os.mkdir(path)
            os.chdir(path)
        os.chdir(home)
        
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),
            },os.path.join(SAVE_PATH, 'checkpoint_epoch_'+str(epoch+1)+'.pt'))
        
        if scheduler: scheduler.step(valid_epoch_loss)
        
        print('{}th Train Loss : {}, Valid Loss : {}'.format(epoch+1, train_epoch_loss, valid_epoch_loss))