# -*- coding:utf-8 -*-
"""
########################################################################################
# Author:chengteng, 2018.01.26                                                         #
# VGG16 implementation in keras with TensorFlow backend                                #
# testing data is Cifar10                                                              #
# Github: https://github.com/ohquai/Image-Classification                               #
# vgg intro: http://www.cs.toronto.edu/~frossard/post/vgg16/                           #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
########################################################################################
"""
import os
import numpy as np

from glob import glob
from shutil import copyfile
from src.model.vgg_bn import Vgg16BN
from src.data.data import read_data
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.datasets import cifar10
from keras.utils import to_categorical
from matplotlib import pyplot as plt


def data_preprocess(x_train, y_train, x_test, y_test):
    # zero-scale for image
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)

    return x_train, y_train, x_test, y_test


def show_model_effect(history):
    # show the data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("Model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")

    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig("Performance:" + str(score[1]) + ".jpg")


# paths
path = 'D:/Project/cifar/cifar10/'
model_path = path + 'models/'
submission_path = path + 'submissions/sub.csv'

# coefficient
n_classes = 10
nb_epoch = 10
batch_size = 64
nb_aug = 5
lr = 0.001

# data
img_width, img_height = 32, 32

# read cifar10 data (from net and from local) and split between train and test sets:
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = read_data(path)

# data pre-process, include scale, categorical etc.
x_train, y_train, x_test, y_test = data_preprocess(x_train, y_train, x_test, y_test)

# get model structure
vgg = Vgg16BN(size=(img_width, img_height), n_classes=n_classes, batch_size=batch_size, lr=lr)
model = vgg.model
model.summary()

info_string = "{0}x{1}_{2}epoch_{3}aug_vgg16-bn".format(img_width, img_height, nb_epoch, nb_aug)
ckpt_fn = model_path + '{val_loss:.2f}-loss_' + info_string + '.h5'

ckpt = ModelCheckpoint(filepath=ckpt_fn, monitor='val_loss', save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history = vgg.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(x_test, y_test), callbacks=[ckpt])
# vgg.fit(train_path, valid_path, nb_trn_samples=nb_train_samples, nb_val_samples=nb_valid_samples, nb_epoch=nb_epoch, callbacks=[ckpt], aug=nb_aug)

score = vgg.evaluate(x_test, y_test, verbose=1)

print("====================================")
print("====================================")
print(score[0])
print(score[1])
print("====================================")
print("====================================")

# save model
# model.save(path+"my_model"+str(score[1])+".h5")

# show model effect
show_model_effect(history)

# # generate predictions
# for aug in range(nb_aug):
#     print("Generating predictions for Augmentation {0}...",format(aug+1))
#     if aug == 0:
#         predictions, filenames = vgg.test(test_path, nb_test_samples, aug=nb_aug)
#     else:
#         aug_pred, filenames = vgg.test(test_path, nb_test_samples, aug=nb_aug)
#         predictions += aug_pred
#
# print("Averaging Predictions Across Augmentations...")
# predictions /= nb_aug
#
# # clip predictions
# c = 0.01
# preds = np.clip(predictions, c, 1-c)
#
# sub_file = submission_path + info_string + '.csv'
#
# with open(sub_file, 'w') as f:
#     print("Writing Predictions to CSV...")
#     f.write('id,label\n')
#     for i, image_name in enumerate(filenames):
#         pred = ['%.6f' % p for p in preds[i, :]]
#         if i % 2500 == 0:
#             print(i, '/', nb_test_samples)
#         f.write('%s,%s\n' % (os.path.basename(image_name).replace('.jpg', ''), (pred[1])))
#     print("Done.")

# 0.25 dropout
# 49984/50000 [============================>.] - ETA: 0s - loss: 0.9163 - acc: 0.6773
# 50000/50000 [==============================] - 256s 5ms/step - loss: 0.9162 - acc: 0.6773 - val_loss: 0.8989 - val_acc: 0.6852

# 没有 dropout
# 49984/50000 [============================>.] - ETA: 0s - loss: 0.7824 - acc: 0.7249
# 50000/50000 [==============================] - 256s 5ms/step - loss: 0.7823 - acc: 0.7249 - val_loss: 0.8750 - val_acc: 0.6970

# 改成maxpool
# 49984/50000 [============================>.] - ETA: 0s - loss: 0.7500 - acc: 0.7376
# 50000/50000 [==============================] - 260s 5ms/step - loss: 0.7500 - acc: 0.7376 - val_loss: 0.8892 - val_acc: 0.6973