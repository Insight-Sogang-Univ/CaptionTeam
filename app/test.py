import os, re, pickle
import itertools

import torch
from torchvision import transforms

from PIL import Image

import numpy as np
import argparse, cv2

from config.test_config import *

def test(img_path, save_path=None, save=True, model='lstm',name=None):
    print(name, "in test.py")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vocab_set = pickle.load(open(MODEL_DIR+'vocab_set.pkl', 'rb')) if os.path.exists(MODEL_DIR+'vocab_set.pkl') else None
    vocab, word2idx, idx2word, max_len = vocab_set
    vocab_size = len(vocab)
    
    #모델 불러오기
    checkpoint = torch.load(os.path.join(MODEL_DIR, MODEL_NAME), map_location=device)
    from models.torch.resnet101_attention import Captioner

    final_model = Captioner(encoded_image_size=14, encoder_dim=2048,
                            attention_dim=ATTENTION_DIM, embed_dim=EMBEDDING_DIM, decoder_dim=DECODER_SIZE,
                            vocab_size=vocab_size,).to(device)
    final_model.load_state_dict(checkpoint['state_dict'])
    final_model.eval()
    
    transformations = transforms.Compose([
    transforms.Resize(256),  # smaller edge of image resized to 256
    transforms.CenterCrop(256),  # get 256x256 crop from random location
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
    ])
    
    image = Image.open(img_path)
    _image = transformations(image).to(device)
    
    capidx = final_model.sample(_image.unsqueeze(0), word2idx['<start>'])
    capidx = capidx[0].detach().cpu().numpy()
    caption=''.join(list(itertools.takewhile(lambda word: word.strip() != '<end>',
                                                            map(lambda idx: idx2word[idx]+' ', iter(capidx)))))

    pronoun = []
    for x in ['he','she','girl','boy','man','woman']:
        pronoun.append(x.capitalize())
        pronoun.append(x.lower())
        
    caption = re.sub('|'.join(pronoun), name, caption)

    # 사진에 캡션 추가
    cap_img = add_caption(image, caption)
    
    cap_img = cv2.cvtColor(cap_img, cv2.COLOR_BGR2RGB)
    
    if save:
        cv2.imwrite(save_path, cap_img)
    else:
        cv2.imshow('imgwithcaption', cap_img)
        cv2.waitKey(0) 

def add_caption(img, caption):
    from PIL import ImageFont, ImageDraw
    
    img_np = np.array(img)
    
    # 캡션 글자 크기 계산
    # size_img = min(img_np.shape[0],img_np.shape[1])
    ft_size = img_np.shape[1]/len(caption) * 1.5
    # ft_size = int(size_img / 14)
    
    ### 캡션 폰트 사이즈 수정
    font=ImageFont.truetype('font/RixYeoljeongdo Regular.ttf',int(ft_size))
    
    draw = ImageDraw.Draw(img) 
    w, h = draw.textsize(caption, font=font, spacing=5)
    img_W = img.size[0]
    img_H = img.size[1]
    
    org=((img_W-w)/2, img_H-(img_H*0.1)) 
    
    bt_img = img_np[:,int(img_H*0.7):,:]
    total_intensity = bt_img.sum()
    avg_intens = total_intensity/(bt_img.shape[0]*bt_img.shape[1])
    if avg_intens>125:
        color = (20,20,20)
    else:
        color = (225,225,225)
    
    draw.text(org,caption,font=font,fill=color)
    
    return np.array(img)

def get_args():
    parser = argparse.ArgumentParser(description = '각종 옵션')
    parser.add_argument('-p', '--image_path', default=r"data\debug\img\bluedragon\bluedragon1.jpg",
                        type=str, help='입력 이미지 경로')
    parser.add_argument('-m', '--model', default='lstm',
                        type=str, help='모델명')
    parser.add_argument('-s', '--save', action='store_true',
                        help='저장?')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    
    test(args.image_path, save_path='sample_image/output.jpg', save=args.save, model=args.model, name='성주')

