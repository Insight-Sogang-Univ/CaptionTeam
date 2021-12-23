 # train.py 에서 저장된 모델을 불러와서, 주어진 이미지에 대해 캡션을 출력해야 함

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image

import argparse
from config import *

from models.encoder_to_decoder import EncodertoDecoder


def get_args():
    parser = argparse.ArgumentParser(description = '각종 옵션')
    parser.add_argument('-lr', '--learning_rate', required=True,
                        type=float, help='모델 이름 입력')
    parser.add_argument('-e', '--epochs', required=True,
                        type=int, help='학습 횟수 입력')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    ### 여기에 내용을 쓰는 거다
    args = get_args()
    
    #사진 받기
    img_path = r"datasets\debug\img\bluedragon\bluedragon1.jpg"
    image = read_image(img_path)
    
    image = image.type(torch.FloatTensor)
    
    transform = transforms.Compose(
        [
        transforms.Resize((299,299)),
        # transforms.ToTensor(),
        transforms.Normalize((127.5, 127.5, 127.5), (255., 255., 255.))
        ]
    )

    image = transform(image)
    
    #모델 불러오기
    model = torch.load(SAVE_PATH, 'checkpoint_epoch_'+str(args.epochs)+'.pt')
    model.eval()
    
    #모델에 사진 넣기
    vocabulary = None
    embedder = None
    caption = model.caption_image(image, vocabulary, embedder, max_length=50)

