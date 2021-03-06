import os
import logging

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

DIR = os.path.abspath(os.path.dirname(__file__))
DATASETS_DICT = {"mnist": "MNIST",
                 "fashion": "FashionMNIST",
                 "yourdata": "YourData"}

DATASETS = list(DATASETS_DICT.keys())

def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))

def get_img_size(dataset):
    """Return the correct image size."""
    return get_dataset(dataset).img_size

def get_num_classes(dataset):
    "Return the number of classes"
    return len(get_dataset(dataset).classes)

def get_class_labels(dataset):
    """Return the class labels"""
    return get_dataset(dataset).classes

def get_dataloaders(dataset, root=None, shuffle=True, is_train=True, pin_memory=True,
                    batch_size=128, logger=logging.getLogger(__name__), **kwargs):
    """A generic data loader
    Parameters
    ----------
    dataset : {"mnist", "fashion"}
        Name of the dataset to load
    root : str
        Path to the dataset root. If `None` uses the default one.
    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    Dataset = get_dataset(dataset)
    dataset = Dataset(is_train=is_train, logger=logger) if root is None else Dataset(is_train=is_train, root=root, logger=logger)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      **kwargs)

class FashionMNIST(datasets.FashionMNIST):
    """Fashion Mnist wrapper. Docs: `datasets.FashionMNIST.`"""
    img_size = (1, 32, 32)

    def __init__(self, root=os.path.join(DIR, '../data/fashionMnist'), is_train=True, **kwargs):
        super().__init__(root,
                         train=is_train,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(32),
                             transforms.ToTensor()
                         ]))

class MNIST(datasets.MNIST):
    """Mnist wrapper. Docs: `datasets.MNIST.`"""
    img_size = (1, 32, 32)

    def __init__(self, root=os.path.join(DIR, '../data/mnist'), is_train=True, **kwargs):
        super().__init__(root,
                         train=is_train,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(32),
                             transforms.ToTensor()
                         ]))

class YourData():

    def __init__(self, root=os.path.join(DIR, '../data/yourdata'), is_train=True, **kwargs):
        pass
