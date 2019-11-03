
if __name__ == '__main__':

    from datasets import CirclesData

    dataset = CirclesData()

    X_train = dataset.X_train  # torch.Tensor containing network entries for training
    X_test = dataset.X_test
    print('Number of train examples:', X_train.size(0))  # affiche la taille des donn´ees : torch.Size([200, 2])
    print('Number of test examples:', X_test.size(0))
    dataset.plot_data()  # plots points in train and test

    # computes predictions y for all points of the grid (forward et params non fournis,
    # à coder)
    # Ygrid = forward(params, data.Xgrid)
    # plots points and decision boundary thanks to the grid
    # data.plot_data_with_grid(Ygrid)

    # plot the curves of loss and accuracy in train and test.
    # data.plot_loss(loss_train, loss_train, acc_train, acc_test)
