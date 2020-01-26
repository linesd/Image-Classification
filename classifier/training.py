
import torch
import logging
from timeit import default_timer

class Trainer():
    """
    Class to handle training of model.
    Parameters
    ----------
    model: disvae.vae.VAE
    optimizer: torch.optim.Optimizer
    loss_f: disvae.models.BaseLoss
        Loss function.
    device: torch.device, optional
        Device on which to run the code.
    logger: logging.Logger, optional
        Logger.
    save_dir : str, optional
        Directory for saving logs.
    gif_visualizer : viz.Visualizer, optional
        Gif Visualizer that should return samples at every epochs.
    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, optimizer, criterion,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results"):

        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.logger = logger
        self.logger.info("Training Device: {}".format(self.device))

    def __call__(self, data_loader,
                 epochs=10):
        """
        Trains the model.
        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        epochs: int, optional
            Number of epochs to train the model for.
        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """

        self.model.train()

        for epoch in range(epochs):
            self._train_epoch(data_loader, epoch)

        self.model.eval()

    def _train_epoch(self, data_loader, epoch):
        """
        Trains the model for one epoch.
        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        storer: dict
            Dictionary in which to store important variables for vizualisation.
        epoch: int
            Epoch number
        Return
        ------
        mean_epoch_loss: float
            Mean loss per image
        """
        epoch_loss = 0.0
        for i, data in enumerate(data_loader):
            # pull the
            inputs, labels = data
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward, backward, optimize
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Print some stats
            epoch_loss += loss.item()
            if i % 200 == 199:  # print every 2000 mini-batches
                print('[Epoch: %d, Num Minibatches: %5d] loss: %.3f' %
                      (epoch + 1, i + 1, epoch_loss / 200))
                epoch_loss = 0.0
