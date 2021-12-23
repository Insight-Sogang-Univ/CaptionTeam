 # train.py 에서 저장된 모델을 불러와서, 주어진 이미지에 대해 캡션을 출력해야 함

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image

import argparse
from config import *

from models.encoder_to_decoder import EncodertoDecoder

import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2


def get_args():
    parser = argparse.ArgumentParser(description = '각종 옵션')
    parser.add_argument('-lr', '--learning_rate', required=False,
                        type=float, help='모델 이름 입력')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    ### 여기에 내용을 쓰는 거다
    args = get_args()
    
    #사진 받기
    img_path = r"datasets\debug\img\bluedragon\bluedragon1.jpg"
    image = read_image(img_path)
    
    _image = image.clone()
    _image = image.type(torch.FloatTensor)
    
    transform = transforms.Compose(
        [
        transforms.Resize((299,299)),
        # transforms.ToTensor(),
        transforms.Normalize((127.5, 127.5, 127.5), (255., 255., 255.))
        ]
    )

    _image = transform(_image)
    
    #모델 불러오기
    # model = torch.load(SAVE_PATH, 'checkpoint_epoch_'+str(args.epochs)+'.pt')
    # model.eval()
    
    #모델에 사진 넣기
    # vocabulary = None
    # embedder = None
    # caption = model.caption_image(image, vocabulary, embedder, max_length=50)
    caption = 'Hello My name is GA-YOUNG.'
    
    #사진에 캡션을 더해서 띄우기 # Open cv 참고
    import matplotlib.pyplot as plt
    # 원 이미지 출력
   
    img = torch.permute(image, (1,2,0))
    img = img.numpy()
    
    img = Image.fromarray(img)
    
    draw = ImageDraw.Draw(img) 
    font=ImageFont.truetype("font/malgun.ttf",15) 
    org=(50,400) 
    draw.text(org,caption,font=font,fill=(225,225,225))

    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('imgwithcaption', img)
    cv2.waitKey(0) 

