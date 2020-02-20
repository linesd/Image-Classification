import os
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from utils.helpers import load_file, load_group
from sklearn.preprocessing import StandardScaler

DIR = os.path.abspath(os.path.dirname(__file__))
DATASETS_DICT = {"har": "HumanActivityRecognition",
                 "newdataset": "NewDataset"}

DATASETS = list(DATASETS_DICT.keys())

def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))

def get_data_size(dataset):
    """Return the correct data size."""
    return get_dataset(dataset).data_size

def get_num_classes(dataset):
    "Return the number of classes"
    return get_dataset(dataset).n_classes

def get_class_labels(dataset):
    """Return the class labels"""
    return get_dataset(dataset).classes

def get_dataloaders(dataset, root=None, shuffle=True, is_train=True, pin_memory=True,
                    batch_size=128, logger=logging.getLogger(__name__), **kwargs):
    """A generic data loader
    Parameters
    ----------
    dataset : {"har"}
        Name of the dataset to load
    root : str
        Path to the dataset root. If `None` uses the default one.
    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    Dataset = get_dataset(dataset)
    dataset = Dataset(is_train=is_train) if root is None else Dataset(is_train=is_train, root=root)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      **kwargs)

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

class HumanActivityRecognition(Dataset):
    """Human activity recognition dataset"""
    data_size = (9, 128)
    n_classes = 6
    classes = ['walking',
               'walking upstairs',
               'walking downstairs',
               'sitting',
               'standing',
               'laying']

    def __init__(self, root=os.path.join(DIR, '../data/UCIHAR/'),
                 is_train=True,
                 standardize=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if is_train:
            image_set = 'train'
        else:
            image_set = 'test'
        print("Loading Human Activity Recognition dataset ...")
        data = self.load_dataset(root, image_set)
        if standardize:
            X = self.scale_data(data[0], standardize=True)
            self.X = torch.from_numpy(X).permute(0,2,1).float()
        else:
            self.X = torch.from_numpy(data[0]).permute(0, 2, 1).float()
        self.Y = torch.from_numpy(data[1]).flatten().long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.X[idx,:,:]
        target = self.Y[idx]

        return input, target

    # load a dataset group, such as train or test
    # borrowed mthods from the tutorial
    def load_dataset_group(self, group, prefix=''):
        filepath = prefix + group + '/Inertial Signals/'
        # load all 9 files as a single array
        filenames = list()
        # total acceleration
        filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
        # body acceleration
        filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
        # body gyroscope
        filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
        # load input data
        X = load_group(filenames, filepath)
        # load class output
        Y = load_file(prefix + group + '/y_' + group + '.txt')
        return X, Y

    # load the dataset, returns train and test X and y elements
    def load_dataset(self, root='', image_set='train'):
        # load all train
        X, Y = self.load_dataset_group(image_set, root)
        # zero-offset class values
        Y = Y - 1
        return X, Y

    # standardize data
    def scale_data(self, X, standardize):
        # remove overlap
        cut = int(X.shape[1] / 2)
        longX = X[:, -cut:, :]
        # flatten windows
        longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
        # flatten train and test
        flatX = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
        # standardize
        if standardize:
            s = StandardScaler()
            # fit on training data
            s.fit(longX)
            # apply to training and test data
            flatX = s.transform(flatX)
        # reshape
        flatX = flatX.reshape((X.shape))
        return flatX



