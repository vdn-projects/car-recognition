import mxnet as mx

# PATH/FILE CONFIG
SQLITE3_DB_FILE = "./db_design/db.sqlite"
FULL_DATASET_FILE = "./db_design/full_dataset.csv"
MISC_FILE_PATH = "./dataset/train_model/misc_files"
TRAIN_IMG_PATH = "./dataset/train_model/car_ims"

TRAIN_LIST_FILE = MISC_FILE_PATH + "/train.lst"
VAL_LIST_FILE = MISC_FILE_PATH + "/val.lst"
TEST_LIST_FILE = MISC_FILE_PATH + "/test.lst"

TRAIN_REC_FILE = MISC_FILE_PATH + "/train.rec"
VAL_REC_FILE = MISC_FILE_PATH + "/val.rec"
TEST_REC_FILE = MISC_FILE_PATH + "/test.rec"

GRAB_LIST_FILE = "./dataset/test_model/misc_files/grab_cars_test.lst"
GRAB_REC_FILE = "./dataset/test_model/misc_files/grab_cars_test.rec"

CHECKPOINT_PATH = "./checkpoints/vggnet"

IM2REC_PY_PATH = "ai4c/lib/python3.6/site-packages/mxnet/tools/im2rec.py"

# NUMBER_CONFIG
LEARNING_RATE = 1e-5
MAX_EPOCH = 150

IMAGE_SIZE = (3, 224, 224)
R_MEAN = 123.68
G_MEAN = 116.779
B_MEAN = 103.939

NUM_CLASSES = 164
NUM_VAL_IMAGES = 0.15
NUM_TEST_IMAGES = 0.15

BATCH_SIZE = 32
NUM_DEVICES = 1

# This is evaluated as the epoch for final model
DEFAULT_EPOCH_NUMBER = 85

# HARDWARE FOR MODEL BUILDING
MODEL_PROCESS_CONTEXT = [mx.gpu(0)]
