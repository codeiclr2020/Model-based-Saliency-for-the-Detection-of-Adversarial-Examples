from torchvision.transforms import *
from torchvision.datasets import ImageFolder
from torch.utils.data import dataloader
from ..utils.pytorch_fixes import *
import os
from PIL import Image

#path files
TRAIN_PATH = './data/train/'
VAL_PATH = './data/val/'


SUGGESTED_BS = 64
NUM_CLASSES = 2
SUGGESTED_EPOCHS_PER_STEP = 11
SUGGESTED_BASE = 64
IMAGE_SIZE = 112

# -----------------------------------------------------

def is_valid_file(path):
    try:
        Image.open(path)
        return True
    except: return False

def get_train_dataset(size=IMAGE_SIZE):
    if not (os.path.exists(TRAIN_PATH) and os.path.exists(VAL_PATH)):
        raise ValueError(
            'Please make sure that you specify a path to the ImageNet dataset folder in sal/datasets/imagenet_dataset.py file!')
    return ImageFolder(TRAIN_PATH, transform=Compose([
        RandomSizedCrop2(size, min_area=0.3),
        RandomHorizontalFlip(),
        ToTensor(),
        STD_NORMALIZE,  # Images will be in range -1 to 1
    ]))


def get_val_dataset(size=IMAGE_SIZE):
    if not (os.path.exists(TRAIN_PATH) and os.path.exists(VAL_PATH)):
        raise ValueError(
            'Please make sure that you specify a path to the ImageNet dataset folder in sal/datasets/imagenet_dataset.py file!')
    return ImageFolder(VAL_PATH, transform=Compose([
        Scale(IMAGE_SIZE),
        CenterCrop(size),
        ToTensor(),
        STD_NORMALIZE,
    ]))

def get_loader(dataset, batch_size=64, pin_memory=True):
    return dataloader.DataLoader(dataset=dataset, batch_size=batch_size,
                                 shuffle=True, drop_last=True, num_workers=8, pin_memory=pin_memory)

SYNSET_TO_NAME = {'cat':'cat', 'dog':'dog'}
SYNSET_TO_CLASS_ID= {'cat':0, 'dog':1}

CLASS_ID_TO_SYNSET = {v:k for k,v in list(SYNSET_TO_CLASS_ID.items())}
CLASS_ID_TO_NAME = CLASS_ID_TO_SYNSET
CLASS_NAME_TO_ID = {v:k for k, v in list(CLASS_ID_TO_NAME.items())}








