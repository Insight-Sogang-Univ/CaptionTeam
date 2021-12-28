import labeling, image_process, naming, argparse

if __package__ is None:
        import sys
        from os import path
        sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
        
from config.train_config import *
# import sys
# sys.path.insert(0,'..')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', required=False, type=int, help='number of images for making hdf5 set')
    args = parser.parse_args()
    
    print('Preprocessing Start')
    
    print('Cleaning names...')
    naming.revising_names(RAW_NAME, NAME_ADD, NAME_DEL, NAME_PATH)
    print('Cleaning Complete')
    
    print('Caption Preprocessing...')
    label_path = labeling.modify_label(RAW_LABEL, DATA_FOLDER, NAME_PATH, num=args.num)
    print('Caption Preprocessing Complete')
    
    print('Images to HDF5...')
    image_process.jpg_to_hdf5(label_path, IMAGE_FILE, RAW_IMAGE, ERROR_LOGS)
    print('HDF5 Created')
    
    print('Trimming Error Images...')
    labeling.clean_data(label_path, ERROR_LOGS)
    print('Error Images Cleaned!')
    
    print('Preprocessing Complete')