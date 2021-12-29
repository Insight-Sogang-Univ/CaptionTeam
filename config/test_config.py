###  CONFIGURATION  ###
DATA_FOLDER= 'data/debug'

###     TEST     ###
MODEL_PATH  = DATA_FOLDER+'/pt/l2/checkpoint_epoch_10.pt' #17, 22, 31, 40
EMBED_PATH  = DATA_FOLDER+'/embedder.pkl'

### EMBEDDING ###
METHOD     = 'fast'
VECTOR_DIM = 256
WINDOW_SIZE= 1
MIN_COUNT  = 2
SG         = 1
FIXED_LENGTH=20