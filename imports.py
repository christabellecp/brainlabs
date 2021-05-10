from __future__ import print_function, division

from CogDataset3d import *
from Smaller_RUNet3d import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import itertools
import os
import time
from pathlib import Path
from tqdm.notebook import tqdm
import math
import pickle
import random
import sys
import os
import gzip
import shutil
import nibabel as nib
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

from scipy.stats import pearsonr
# pytorch stuff
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pickle

# need for AMP
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

torch.manual_seed(3)
from tqdm.notebook import tqdm


from PIL import Image
import torch.utils.data
from skimage import io
from torch.utils.data import Dataset
import random



from   category_encoders             import *

from   sklearn.compose               import *
from   sklearn.ensemble              import *
from   sklearn.linear_model          import *
from   sklearn.impute                import *
from   sklearn.metrics               import *
from   sklearn.pipeline              import *
from   sklearn.preprocessing         import *
from   sklearn.model_selection       import *
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
import lazypredict
from lazypredict.Supervised import LazyRegressor

import nibabel as nib
from joblib import dump, load
np.set_printoptions(threshold=sys.maxsize)

