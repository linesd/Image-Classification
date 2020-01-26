import torch
from torch import nn

class LeNet5(nn.Module):
    def __init__(self, img_size, n_classes):
        """
         Input - 1x32x32
         C1 - 6@28x28 (5x5 kernel)
         tanh
         S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
         C3 - 16@10x10 (5x5 kernel, complicated shit)
         tanh
         S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
         C5 - 120@1x1 (5x5 kernel)
         F6 - 84
         tanh
         F7 - 10 (Output)
        """
        super(LeNet5, self).__init__()
        self.n_chan = img_size[0]
        self.n_classes = n_classes

        # Convolutional Layers
        self.conv1 = nn.Conv2d(self.n_chan, 6, kernel_size=(5, 5), stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5, 5), stride=1)

        # Fully connected layers
        self.lin1 = nn.Linear(120, 84)
        self.lin2 = nn.Linear(84, self.n_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))

        # Fully connected layers
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))

        return x
