import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import warnings

warnings.filterwarnings("ignore")

from utils import config
from train import SegNetTrain
from pytorch_lightning import seed_everything


if __name__ == '__main__':
    seed_everything(233)
    args = config()
    SegNetTrain(args, 'srscn')