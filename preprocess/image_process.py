import h5py
import argparse
import pandas as pd

from tqdm import tqdm
from torchvision.io import read_image

def jpg_to_hdf5(csv_path, save_path, img_path, error_path='error_log.txt'):
    df = pd.read_csv(csv_path)
    
    with h5py.File(save_path, 'w') as hdf:
        for i, folder in enumerate(df['head'].unique()):
            print(f"Saving {i+1}/{len(df['head'].unique())}")
            f = hdf.create_group(folder)
            for img in tqdm(df.loc[df['head']==folder,'name'].values):
                try:
                    image = read_image(img_path+'/'+folder+'/'+img)
                except:
                    with open(error_path, 'a') as e:
                        e.write(img)
                        e.write('\n')
                    continue
                
                if image.shape[0]==4:
                    image = image[:3]
                    
                f.create_dataset(img, data=image, compression='gzip')

def get_args():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('-f','--img_path', required=True,
                        type=str, help='Image File path')
    parser.add_argument('-s','--save_path', required=False,
                        type=str, help='File path for Saving')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = get_args()
    OUTPUT  = '../data'
    path = OUTPUT+'/captions_processed.csv'
    
    save_path = args.save_path+'/images.hdf5' if args.save_path else args.img_path+'/images.hdf5'
    
    jpg_to_hdf5(path, save_path, args.img_path)