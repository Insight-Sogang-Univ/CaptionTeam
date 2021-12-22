###  CONFIGURATION  ###

###    PATH   ###
LABEL_PATH = 'datasets/debug/captions.csv'
IMAGE_PATH = 'datasets/debug/img'
EMBED_PATH = 'data/embedder.pkl'
DIC_PATH   = 'C:\mecab\mecab-ko-dic'

### EMBEDDING ###
METHOD     = 'fast'
VECTOR_DIM = 256
WINDOW_SIZE= 1
MIN_COUNT  = 2
SG         = 1

#### TRAIN ####
BATCH_SIZE = 64

SAVE_PATH  = 'pt'