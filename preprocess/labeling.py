import re, os
import sys
sys.path.insert(0,'..')

import argparse
import pandas as pd

import naming

def modify_label(csv_path, new_path, names, name='홍길동'):
    
    captions = pd.read_csv(csv_path)
    names = pd.read_csv(names)

    # 폴더명 틀려서 수정
    for index in captions[captions['head']=='MarieClaire_award'].index:
        captions.loc[index,'head'] = 'MarieClaire_awards'
        captions.loc[index,'name'] = captions.loc[index,'name'][:17]+'s'+captions.loc[index,'name'][17:]

    # 외국인 위주 데이터 제거
    overseas = ['academy', 'cannes', 'fawards', 'fmovie', 'fmusic', 'oversea_celebrity']
    captions = captions[~captions['head'].isin(overseas)].reset_index(drop=True)
    
    # 이름을 '홍길동으로 대체'
    captions = naming.replace_names(captions, names, repName=name)
    
    # HD_photo 수정
    HD_photos = captions.loc[captions['head']=='HD_photo','caption']
    captions.loc[captions['head']=='HD_photo','caption'] = HD_photos.apply(lambda x: re.search('(.*(?=ㅣ))|(.*(?!=ㅣ)(?=$))', x).group())
    
    new_file = 'captions.csv'
    i = 0
    while os.path.exists(new_path+'/'+new_file):
        new_file = f'captions_{i}.csv'
        i += 1
        
    captions.to_csv(new_path+'/'+new_file, index=False)
    
    return new_path+'/'+new_file

def clean_data(csv_path, error_path):
    df = pd.read_csv(csv_path)
    
    # try:
    with open(error_path, 'r') as f:
        errors = f.readlines()
    errors = [error[:-1] for error in errors]
    df = df[~df['name'].isin(errors)]
    df.to_csv(csv_path, index=False)
    # except:
    #     pass

def get_args():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('-nf','--names', required=True,
                        type=str, help='Name File path')
    parser.add_argument('-n','--name', default='홍길동',
                        type=str, help='New Name For Replace')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    
    OUTPUT  = '../data'
    path = OUTPUT + '/captions_raw.csv'
    
    modify_label(path, OUTPUT, args.names, args.name)