
import matplotlib.pyplot as plt
plt.ion()
import torch
import numpy as np
import scipy.io

class CirclesData:
    """Simple dataset of circular data with two classes"""
    def __init__(self):
        # Grid
        x1, x2 = np.meshgrid(np.arange(-2, 2, 0.1), np.arange(-2, 2, 0.1))
        self._X_grid = np.array([x1.flatten(), x2.flatten()]).T.astype('float32')

        # Real data
        circles = scipy.io.loadmat("data/circles.mat")
        self._X_train = circles["Xtrain"].astype('float32')
        self._X_test = circles["Xtest"].astype('float32')
        self._y_train = circles["Ytrain"].astype('float32')
        self._y_test = circles["Ytest"].astype('float32')

        self._X_grid_th = torch.from_numpy(self._X_grid)
        self._X_train_th = torch.from_numpy(self._X_train)
        self._X_test_th = torch.from_numpy(self._X_test)
        self._y_train_th = torch.from_numpy(self._y_train)
        self._y_test_th = torch.from_numpy(self._y_test)

        self.loss_train = []
        self.loss_test = []
        self.acc_train = []
        self.acc_test = []

    def __getattr__(self, key):
        if key == "X_grid": return self._X_grid_th
        if key == "X_train": return self._X_train_th
        if key == "X_test": return self._X_test_th
        if key == "y_train": return self._y_train_th
        if key == "y_test": return self._y_test_th
        return None

    def plot_data(self):
        plt.figure(1, figsize=(5,5))
        plt.plot(self._X_train[self._y_train[:,0] == 1,0], self._X_train[self._y_train[:,0] == 1,1], 'bo', label='Train')
        plt.plot(self._X_train[self._y_train[:,1] == 1,0], self._X_train[self._y_train[:,1] == 1,1], 'ro')
        plt.plot(self._X_test[self._y_test[:,0] == 1,0], self._X_test[self._y_test[:,0] == 1,1], 'b+', label="Test")
        plt.plot(self._X_test[self._y_test[:,1] == 1,0], self._X_test[self._y_test[:,1] == 1,1], 'r+')
        plt.legend()
        plt.show()

    def plot_data_with_grid(self, y_grid, title=""):
        plt.figure(2)
        y_grid = y_grid[:,1].numpy()
        plt.clf()
        plt.imshow(np.reshape(y_grid, (40,40)))
        plt.plot(self._X_train[self._y_train[:,0] == 1,0]*10+20, self._X_train[self._y_train[:,0] == 1,1]*10+20, 'bo', label="Train")
        plt.plot(self._X_train[self._y_train[:,1] == 1,0]*10+20, self._X_train[self._y_train[:,1] == 1,1]*10+20, 'ro')
        plt.plot(self._X_test[self._y_test[:,0] == 1,0]*10+20, self._X_test[self._y_test[:,0] == 1,1]*10+20, 'b+', label="Test")
        plt.plot(self._X_test[self._y_test[:,1] == 1,0]*10+20, self._X_test[self._y_test[:,1] == 1,1]*10+20, 'r+')
        plt.xlim(0,39)
        plt.ylim(0,39)
        plt.clim(0.3,0.7)
        plt.title(title)
        plt.draw()
        plt.pause(1e-3)

    def plot_loss_accuracy(self, loss_train, loss_test, acc_train, acc_test, ax=None):

        self.loss_train.append(loss_train)
        self.loss_test.append(loss_test)
        self.acc_train.append(acc_train)
        self.acc_test.append(acc_test)
        plt.figure(3)
        plt.clf()
        plt.subplot(1,2,1)
        plt.plot(np.array(self.acc_train), label="Acc. Train")
        plt.plot(np.array(self.acc_test), label="Acc. Test")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(np.array(self.loss_train), label="Loss Train")
        plt.plot(np.array(self.loss_test), label="Loss Test")
        plt.legend()
        plt.show()
        plt.pause(1e-3)

class MNISTData:
    """MNIST Dataset of hand-written digits"""
    def __init__(self):
        # Real data
        mnist = scipy.io.loadmat("data/mnist.mat")
        self._X_train_th = torch.from_numpy(mnist["Xtrain"].astype('float32'))
        self._X_test_th = torch.from_numpy(mnist["Xtest"].astype('float32'))
        self._y_train_th = torch.from_numpy(mnist["Ytrain"].astype('float32'))
        self._y_test_th = torch.from_numpy(mnist["Ytest"].astype('float32'))

    def __getattr__(self, key):
        if key == "X_train": return self._X_train_th
        if key == "X_test": return self._X_test_th
        if key == "y_train": return self._y_train_th
        if key == "y_test": return self._y_test_th
        return None
