
if __name__ == '__main__':

    from tme5 import CirclesData

    dataset = CirclesData()

    Xtrain = dataset.Xtrain  # torch.Tensor containing network entries for training

    print(Xtrain.size())  # affiche la taille des donn´ees : torch.Size([200, 2])
    N = Xtrain.size(0)  # number of example
    nx = Xtrain.size(1)  # dimensionality of input

    dataset.plot_data()  # plots points in train and test
    # computes predictions y for all points of the grid (forward et params non fournis,
    # à coder)
    # Ygrid = forward(params, data.Xgrid)
    # plots points and decision boundary thanks to the grid
    # data.plot_data_with_grid(Ygrid)

    # plot the curves of loss and accuracy in train and test.
    # data.plot_loss(loss_train, loss_train, acc_train, acc_test)
