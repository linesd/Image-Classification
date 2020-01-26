import argparse
import sys
import os

from torch import optim
from torch import nn

from utils.datasets import get_dataloaders, get_img_size, get_num_classes, DATASETS
from utils.helpers import get_config_section, FormatterNoDuplicate, set_seed, create_safe_directory
from classifier.cnn import init_specific_model
from classifier.training import Trainer
from classifier.utils.modelIO import save_model

from classifier.cnn import MODELS

CONFIG_FILE = "hyperparams.ini"
RES_DIR = "results"

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

    # Model Options
    model = parser.add_argument_group('Model specific options')
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=MODELS,
                       help='Type of encoder to use.')

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('-n', '--name', type=str, default=default_config['name'],
                         help="Name of the model for storing and loading purposes.")
    general.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')

    args = parser.parse_args(args_to_parse)

    return args


def main(args):
    """Main train and evaluation function.
        Parameters
        ----------
        args: argparse.Namespace
            Arguments
    """

    set_seed(args.seed)
    exp_dir = os.path.join(RES_DIR, args.name)

    # Create directory (if same name exists, archive the old one)
    create_safe_directory(exp_dir)

    # PREPARES DATA
    train_loader = get_dataloaders(args.dataset,
                               batch_size=args.batch_size)

    # PREPARES MODEL
    args.img_size = get_img_size(args.dataset)
    args.num_classes = get_num_classes(args.dataset)
    model = init_specific_model(args.model_type, args.img_size, args.num_classes)

    # TRAINS
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion)
    trainer(train_loader, args.epochs)

    # SAVE MODEL AND EXPERIMENT INFORMATION
    save_model(trainer.model, exp_dir, metadata=vars(args))



if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)