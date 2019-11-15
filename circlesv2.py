
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

from datasets import CirclesData


class Net():
    """One-hidden layer perceptron.

    Args:
    in_size (int): number of input features
    out_classes (int): number of hidden cells
    out_classes (int): number of output classes
    """
    def __init__(self, in_size, hidden_size, out_classes):

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_classes = out_classes
        # initialize weights
        self.init_params()

    def init_params(self):
        """Initialize the parameters of network.
        Values are initialized from \mathcal{U}(-\sqrt{k}, \sqrt{k})),
        where k = \frac{1}{\text{in\_size}}
        """
        params = {}
        k = math.sqrt(1 / in_size)
        self.Wh = torch.rand(self.hidden_size, self.in_size)* 2 * k - k
        self.bh = torch.rand(self.hidden_size, 1)* 2 * k - k
        self.Wy = torch.rand(self.out_classes, self.hidden_size)* 2 * k - k
        self.by = torch.rand(self.out_classes, 1)* 2 * k - k
        return params

    def forward(self, x):
        """Forward pass to compute intermediary steps and output.
        Args:
            x: input batch of size (B, Nx)
        """
        htilde = torch.matmul(x, self.Wh.T) + self.bh.T
        h = torch.tanh(htilde)
        ytilde = torch.matmul(h, self.Wy.T) + self.by.T
        yexp = torch.exp(ytilde)
        yhat = yexp / yexp.sum(1, keepdim=True)

        # return the intermediary steps and the output yhat
        outputs = {'x': x, 'htilde': htilde, 'h': h, 'ytilde': ytilde, 'yhat': yhat}
        return yhat, outputs

    def backward(self, outputs, y):
        """Backward pass to calculate gradients w.r.t parameters.
        Args:
            outputs (dict): tensors outputs from forward
            y (Tensor): matrix of targets, each row is a one-hot encoded vector
        """
        yhat = outputs['yhat']
        h = outputs['h']
        x = outputs['x']

        # backward pass
        grad_ytilde = yhat - y
        grad_Wy = torch.matmul(grad_ytilde.T, h)
        grad_by = grad_ytilde.T.sum(1, keepdim=True)
        grad_h = torch.matmul(grad_ytilde, self.Wy)
        grad_htilde = grad_h * (1 - h**2)

        grad_Wh = torch.matmul(grad_htilde.T, x)
        grad_bh = grad_htilde.T.sum(1, keepdim=True)

        # set the gradients as attributes
        self.grad_Wh = grad_Wh
        self.grad_bh = grad_bh
        self.grad_Wy = grad_Wy
        self.grad_by = grad_by

    def parameters(self):
        return (self.Wh, self.bh, self.Wy, self.by)

    def gradients(self):
        return (self.grad_Wh, self.grad_bh, self.grad_Wy, self.grad_by)


def cross_entropy_loss(yhat, y):
    """Cross-entropy loss.
    This function assumes yhat and y are batches of labels (size (B, C)).
    Args:
        yhat (Tensor): matrix of predictions, each row sums to 1
        y (Tensor): matrix of targets, each row is one-hot encoded
    """
    loss = - torch.sum(y * torch.log(yhat), 1)
    return torch.mean(loss)


def accuracy(yhat, y):
    """Accuracy: the rate of correct predictions.
    Args:
        yhat (Tensor): matrix of predictions, each row sums to 1
        y (Tensor): matrix of targets, each row is a one-hot encoded vector
    """
    _, yhat_inds = torch.max(yhat, 1)

    # one-hot encode the predictions
    # yhat_hot = torch.zeros_like(yhat)
    # yhat_hot.scatter_(1, inds.view(-1,1), 1)
    # acc = (pred_hot == y).to(torch.float).mean()

    # extract indices of labels in y
    _, y_inds = torch.max(y, 1)
    acc = (yhat_inds == y_inds).to(torch.float).mean().item()
    return acc


def sgd_step(net, eta):
    """One parameter update of SGD"""
    net.Wh -= eta * net.grad_Wh
    net.bh -= eta * net.grad_bh
    net.Wy -= eta * net.grad_Wy
    net.by -= eta * net.grad_by

def train(dataset, net, epochs, eta=0.03, savedir=None):
    """Trains a one-hidden layer perceptron on the Circles dataset."""

    X_train = dataset.X_train
    y_train = dataset.y_train
    X_test = dataset.X_test
    y_test = dataset.y_test

    for epoch in range(1, epochs+1):

        yhat, outputs = net.forward(X_train)
        loss = cross_entropy_loss(yhat, y_train)  # train loss
        net.backward(outputs, y_train)

        sgd_step(net, eta)

        acc = accuracy(yhat, y_train)  # train accuracy

        yhat_t, outputs_t = net.forward(X_test)
        loss_t = cross_entropy_loss(yhat_t, y_test)  # test loss
        acc_t = accuracy(yhat_t, y_test)

        # plot loss and accuracy history
        dataset.plot_loss_accuracy(loss.item(), loss_t.item(), acc, acc_t,
                                   savepath=savedir/'Training-logs.pdf' if savedir else None)

        # plot decision boundary of the current model
        yhat_grid, outputs = net.forward(dataset.X_grid)
        dataset.plot_data_with_grid(yhat_grid, f"Decision Boundary (Epoch {epoch})",
                                    savepath=savedir/'Decision-boundary.pdf' if savedir else None)

if __name__ == '__main__':

    from pathlib import Path

    # unit_test0()
    dataset = CirclesData()
    in_size = 2  # input features
    hidden_size = 10  # hidden cells
    out_classes = 2  # classes
    epochs = 100
    eta = 0.006  # learning rate

    net = Net(in_size, hidden_size, out_classes)

    savedir = Path('./figs/torch-linear-init/')
    train(dataset, net, epochs, eta, savedir)

    X_train = dataset.X_train
    X_test = dataset.X_test

    yhat_grid, outputs = net.forward(dataset.X_grid)
    dataset.plot_data_with_grid(yhat_grid, title="Decision Boundary")
