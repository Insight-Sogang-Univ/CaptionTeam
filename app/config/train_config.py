###  CONFIGURATION  ###
DATA_FOLDER= 'data/debug'

###   PREPROCESS   ###
RAW_LABEL  = DATA_FOLDER+'/captions_raw.csv'
RAW_IMAGE  = DATA_FOLDER+'/img'
RAW_NAME   = 'preprocess/names_raw.csv'
NAME_ADD   = 'preprocess/names_for_add.txt'
NAME_DEL   = 'preprocess/names_for_delete.txt'
NAME_PATH  = 'preprocess/names.csv'
ERROR_LOGS = 'preprocess/error_logs.txt'

###    PATH   ###
LABEL_PATH = DATA_FOLDER+'/captions.csv'
EMBED_LABEL= DATA_FOLDER+'/captions_processed.csv'
IMAGE_FILE = DATA_FOLDER+'/images.hdf5'
EMBED_PATH = DATA_FOLDER+'/embedder.pkl'
DIC_PATH   = 'C:\mecab\mecab-ko-dic'

### EMBEDDING ###
METHOD     = 'fast'
VECTOR_DIM = 256
WINDOW_SIZE= 1
MIN_COUNT  = 2
SG         = 1
FIXED_LENGTH=20

#### TRAIN ####
BATCH_SIZE = 64
VALID_RATE = 0.2
SAVE_PATH  = DATA_FOLDER+'/pt'