 # train.py 에서 저장된 모델을 불러와서, 주어진 이미지에 대해 캡션을 출력해야 함

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_image
from PIL import ImageFont, ImageDraw, Image

import numpy as np
import argparse, cv2

from config.test_config import *
from datasets.embedding import CaptionEmbedder

def test(img_path, save_path=None, save=True):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    image = read_image(img_path)
    
    _image = image.clone().to(device)
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
    model = torch.load(MODEL_PATH).to(device)
    model.eval()
    
    #모델에 사진 넣기
    embedder = CaptionEmbedder.load(EMBED_PATH)
    caption = ''.join(model.caption_image(_image, embedder, max_length=FIXED_LENGTH))
    
    #사진에 캡션 추가
    img = torch.permute(image, (1,2,0))
    img = img.numpy()
    
    img = Image.fromarray(img)
    
    draw = ImageDraw.Draw(img) 
    font=ImageFont.truetype("font/malgun.ttf",15)
    
    w, h = draw.textsize(caption, font=font)
    img_W = img.size[0]
    img_H = img.size[1]
    
    org=((img_W-w)/2, img_H-(img_H*0.1)) 
    draw.text(org,caption,font=font,fill=(225,225,225))

    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if save:
        cv2.imwrite(save_path, img)
    else:
        cv2.imshow('imgwithcaption', img)
        cv2.waitKey(0) 

def get_args():
    parser = argparse.ArgumentParser(description = '각종 옵션')
    parser.add_argument('-p', '--image_path', default=r"data\debug\img\bluedragon\bluedragon1.jpg",
                        type=str, help='입력 이미지 경로')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    
    test(args.image_path, save=False)

