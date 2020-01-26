import argparse
import logging
import sys

from torch import optim
from torch import nn

from utils.datasets import get_dataloaders, get_img_size, get_num_classes, DATASETS
from utils.helpers import (get_config_section, FormatterNoDuplicate)
from classifier.models import LeNet5
from classifier.training import Trainer

CONFIG_FILE = "hyperparams.ini"

def parse_arguments(args_to_parse):
    """Parse the command line arguments.
        Parameters
        ----------
        args_to_parse: list of str
            Arguments to parse (split on whitespaces).
    """
    description = "PyTorch implementation of convolutional neural network for image classification"
    default_config = get_config_section([CONFIG_FILE], "Custom")
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default=default_config['dataset'], choices=DATASETS)
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='Learning rate.')
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')

    args = parser.parse_args(args_to_parse)

    return args


def main(args):
    """Main train and evaluation function.
        Parameters
        ----------
        args: argparse.Namespace
            Arguments
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    # logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    # stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    # PREPARES DATA
    train_loader = get_dataloaders(args.dataset,
                               batch_size=args.batch_size,
                               logger=logger)

    # PREPARES MODEL
    args.img_size = get_img_size(args.dataset)
    args.num_classes = get_num_classes(args.dataset)
    model = LeNet5(args.img_size, args.num_classes)

    # TRAINS
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, criterion)
    trainer(train_loader, args.epochs)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)