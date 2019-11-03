
import torch
import numpy as np
import matplotlib.pyplot as plt

from datasets import CirclesData

def init_params(nx, nh, ny):
    """nx: number of input features, nh: number of hidden cells,
    ny: number of output classes"""

    params = {}
    # fill with parameters Wh, Wy, bh, by
    params['Wh'] = torch.randn(nh, nx) * 0.3
    params['bh'] = torch.randn(nh, 1) * 0.3
    params['Wy'] = torch.randn(ny, nh) * 0.3
    params['by'] = torch.randn(ny, 1) * 0.3
    return params


def forward(params, x):

    # computer intermediary steps and output
    results = {}
    Wh = params['Wh']
    bh = params['bh']
    Wy = params['Wy']
    by = params['by']

    B, Nx = tuple(x.size())
    Ny, Nh = tuple(Wy.size())

    htilde = torch.matmul(x, Wh.T) + bh.T
    # htilde = torch.matmul(params['Wh'], x) + params['bh']
    # -> (B, Nh)
    assert(htilde.size() == (B, Nh))
    h = torch.tanh(htilde)
    # -> (B, Nh)
    ytilde = torch.matmul(h, Wy.T) + by.T
    assert(ytilde.size() == (B, Ny))

    # ytilde = torch.matmul(params['Wy'], h) + params['by']
    # -> (B, Ny)
    yexp = torch.exp(ytilde)
    yhat = yexp / yexp.sum(1, keepdim=True)
    assert(yhat.size() == (B, Ny))
    # -> (B, Ny)
    # return the intermediary steps and the output yhat
    results = {'x': x, 'htilde': htilde, 'h': h, 'ytilde': ytilde, 'yhat': yhat}
    return yhat, results

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
    """Accuracy: the rate of correct predictions
    Args:
        yhat (Tensor): matrix of predictions, each row sums to 1
        y (Tensor): matrix of targets, each row is one-hot encoded
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


def backward(params, outputs, y):

    grads = {}
    yhat = outputs['yhat']
    h = outputs['h']
    Wy = params['Wy']
    x = outputs['x']

    B, Nx = tuple(x.size())
    Ny, Nh = tuple(Wy.size())

    # backward pass
    grad_ytilde = yhat - y
    assert(grad_ytilde.size() == (B, Ny))

    grad_Wy = torch.matmul(grad_ytilde.T, h)
    assert(grad_Wy.size() == (Ny, Nh))
    grad_by = grad_ytilde.T.sum(1, keepdim=True)
    assert(grad_by.size() == (Ny, 1))

    grad_h = torch.matmul(grad_ytilde, Wy)
    grad_htilde = grad_h * (1 - h**2)

    grad_Wh = torch.matmul(grad_htilde.T, x)
    assert(grad_Wh.size() == (Nh, Nx))
    grad_bh = grad_htilde.T.sum(1, keepdim=True)
    assert(grad_bh.size() == (Nh, 1))

    # return gradients with respects to parameters
    grads = {'Wh': grad_Wh, 'bh': grad_bh, 'Wy': grad_Wy, 'by': grad_by}
    return grads

def sgd_step(params, grads, eta):
    # TODO mettre à jour le contenu de params
    for key in params:
        params[key] -= eta * grads[key]
    return params



def train(dataset, nx, nh, ny, epochs, eta=0.03):

    params = init_params(nx, nh, ny)

    X_train = dataset.X_train
    y_train = dataset.y_train
    X_test = dataset.X_test
    y_test = dataset.y_test

    for epoch in range(epochs):

        yhat, outputs = forward(params, X_train)
        loss = cross_entropy_loss(yhat, y_train)
        grads = backward(params, outputs, y_train)

        params = sgd_step(params, grads, eta)

        acc = accuracy(yhat, y_train)
        yhat_t, outputs_t = forward(params, X_test)
        loss_t = cross_entropy_loss(yhat_t, y_test)
        acc_t = accuracy(yhat_t, y_test)
        dataset.plot_loss_accuracy(loss.item(), loss_t.item(), acc, acc_t)

    return params


def unit_test0():

    # init
    dataset = CirclesData()
    B = dataset.X_train.shape[0]
    batch_size = 8
    nx = dataset.X_train.shape[1]
    nh = 10
    ny = dataset.y_train.shape[1]
    eta = 0.03

    X = dataset.X_train[:batch_size]
    y = dataset.y_train[:batch_size]
    # Premiers tests, code à modifier
    params = init_params(nx, nh, ny)
    print('Parameters sizes:', {key: tuple(params[key].size())
                                for key in params.keys()})

    yhat, outputs = forward(params, X)
    print(f'yhat:\n{yhat}')
    loss = cross_entropy_loss(yhat, y)
    print('Loss:', loss.item())
    acc = accuracy(yhat, y)
    print("Accuracy", acc)

    grads = backward(params, outputs, y)
    print('Grad sizes:', {key: tuple(grads[key].size())
                          for key in grads.keys()})
    params = sgd_step(params, grads, eta)
    print('Parameters update sucessful')

if __name__ == '__main__':

    # unit_test0()
    dataset = CirclesData()
    nx = 2
    nh = 10
    ny = 2
    epochs = 1000
    eta = 0.03
    params = train(dataset, nx, nh, ny, epochs, eta)

    X_train = dataset.X_train
    X_test = dataset.X_test
    yhat, outputs = forward(params, X_train)
    # Wait for a key to be pressed to keep the figures
    # input("Done")
