# Image-Classification

This repository is a PyTorch implementation of an image classifier based on the LeNet-5 CNN architecture.

 **Notes:**
- Tested for python >= 3.6
- Only tested for CPU

**Table of Contents:**
1. [Install](#install)
2. [Run](#run)
3. [Data](#data)
4. [Results](#results)

## Install

```
# clone repo
pip install -r requirements.txt
```

## Run

Use `python main.py` to run the preset configuration to train and evaluate the model. The preset 
configuration can be found in `hyperparams.ini`.

To run a custom experiment use `python main.py <experiment name> <params>`. For example:

```
python main.py -n test_fashion_1 -d fashion -b 32 --lr 0.0001 
```

You can evaluate a pre-trained model with the following:

```
python main.py -n test_fashion_1 --is-eval-only
```

### Output
Running will create a directory `results/<saving-name>/` which contains:
* **model.pt**: The trained model.
* **specs.json**: The parameters used to run the program (default and those modified with the CLI)

### Help
```
usage: main.py [-h] [-d {mnist,fashion}] [-b BATCH_SIZE] [--lr LR] [-e EPOCHS]
               [-m {Lenet5}] [-n NAME] [-s SEED] [--is-eval-only] [--no-test]

PyTorch implementation of convolutional neural network for image
classification

optional arguments:
  -h, --help            show this help message and exit

Training specific options:
  -d, --dataset {mnist,fashion}
                        Path to training data. (default: fashion)
  -b, --batch-size BATCH_SIZE
                        Batch size for training. (default: 64)
  --lr LR               Learning rate. (default: 0.0005)
  -e, --epochs EPOCHS   Maximum number of epochs to run for. (default: 15)

Model specific options:
  -m, --model-type {Lenet5}
                        Type of encoder to use. (default: LeNet5)

General options:
  -n, --name NAME       Name of the model for storing and loading purposes.
                        (default: fashion_1)

Evaluation specific options:
  --is-eval-only        Whether to only evaluate using precomputed model
                        `name`. (default: False)
  --no-test             Whether or not to compute the test losses.` (default:
                        False)
```

## Data

Current datasets that can be used (these will download by themselves):
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)

### Adding your data

To use your own data you should package it as a PyTorch `dataset` constructor. For more information see the link:
- [dataset](https://pytorch.org/docs/stable/data.html)

You can do this by adding your data to the `YourData` class at the bottom of the file `datasets.py`. 

## Results

Pre-trained models for `fashionMNIST` and `MNIST` can be found in the results folder. 

The following results were achieved on the `fashionMNIST` dataset:
- Epochs: 15
- learning rate: 5e-4
- batch_size: 64

```
***************************************************
*            Evaluating Train Accuracy            *
***************************************************

Train accuracy of the network on the 60000 test images: 98 %

Accuracy of 0 - zero : 96 %
Accuracy of 1 - one : 100 %
Accuracy of 2 - two : 95 %
Accuracy of 3 - three : 98 %
Accuracy of 4 - four : 98 %
Accuracy of 5 - five : 99 %
Accuracy of 6 - six : 94 %
Accuracy of 7 - seven : 99 %
Accuracy of 8 - eight : 99 %
Accuracy of 9 - nine : 97 %

***************************************************
*            Evaluating Test Accuracy             *
***************************************************

Test accuracy of the network on the 10000 test images: 91 %

Accuracy of 0 - zero : 86 %
Accuracy of 1 - one : 98 %
Accuracy of 2 - two : 90 %
Accuracy of 3 - three : 89 %
Accuracy of 4 - four : 89 %
Accuracy of 5 - five : 98 %
Accuracy of 6 - six : 76 %
Accuracy of 7 - seven : 99 %
Accuracy of 8 - eight : 96 %
Accuracy of 9 - nine : 91 %
```

And for the `MNIST` dataset:
- Epochs: 10
- learning rate: 5e-4
- batch_size: 64

```
***************************************************
*            Evaluating Train Accuracy            *
***************************************************

Train accuracy of the network on the 60000 test images: 99 %

Accuracy of 0 - zero : 100 %
Accuracy of 1 - one : 100 %
Accuracy of 2 - two : 99 %
Accuracy of 3 - three : 99 %
Accuracy of 4 - four : 99 %
Accuracy of 5 - five : 99 %
Accuracy of 6 - six : 100 %
Accuracy of 7 - seven : 99 %
Accuracy of 8 - eight : 99 %
Accuracy of 9 - nine : 99 %

***************************************************
*            Evaluating Test Accuracy             *
***************************************************

Test accuracy of the network on the 10000 test images: 99 %

Accuracy of 0 - zero : 100 %
Accuracy of 1 - one : 100 %
Accuracy of 2 - two : 99 %
Accuracy of 3 - three : 98 %
Accuracy of 4 - four : 98 %
Accuracy of 5 - five : 100 %
Accuracy of 6 - six : 100 %
Accuracy of 7 - seven : 99 %
Accuracy of 8 - eight : 99 %
Accuracy of 9 - nine : 98 %
```