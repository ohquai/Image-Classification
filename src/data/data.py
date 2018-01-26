# -*- coding:utf-8 -*-
import pickle
import numpy as np


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        # datadict = p.load(f)
        datadict = pickle.load(f, encoding = 'bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


def read_data(PATH):
    imgX1, imgY1 = load_CIFAR_batch(PATH + 'data_batch_1')
    imgX2, imgY2 = load_CIFAR_batch(PATH + 'data_batch_2')
    imgX3, imgY3 = load_CIFAR_batch(PATH + 'data_batch_3')
    imgX4, imgY4 = load_CIFAR_batch(PATH + 'data_batch_4')
    imgX5, imgY5 = load_CIFAR_batch(PATH + 'data_batch_5')
    imgX = np.concatenate([imgX1, imgX2, imgX3, imgX4, imgX5], axis=0)
    imgY = np.concatenate([imgY1, imgY2, imgY3, imgY4, imgY5], axis=0)
    imgXt, imgYt = load_CIFAR_batch(PATH + 'test_batch')
    print(imgX.shape)
    print(imgY.shape)
    return (imgX, imgY), (imgXt, imgYt)


# if __name__ == "__main__":
#     PATH = 'D:/Project/cifar/cifar10/'
#     imgX1, imgY1 = load_CIFAR_batch(PATH + 'data_batch_1')
#     imgX2, imgY2 = load_CIFAR_batch(PATH + 'data_batch_2')
#     imgX3, imgY3 = load_CIFAR_batch(PATH + 'data_batch_3')
#     imgX4, imgY4 = load_CIFAR_batch(PATH + 'data_batch_4')
#     imgX5, imgY5 = load_CIFAR_batch(PATH + 'data_batch_5')
#
#     imgX = np.concatenate([imgX1, imgX2, imgX3, imgX4, imgX5], axis=0)
#     imgY = np.concatenate([imgY1, imgY2, imgY3, imgY4, imgY5], axis=0)
#
#     imgXt, imgYt = load_CIFAR_batch(PATH + 'test_batch')
#
#     print(imgX.shape)
#     print(imgY.shape)
