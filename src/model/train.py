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
from keras.callbacks import Callback, TensorBoard
import tensorflow as tf
from matplotlib import pyplot as plt


# 构建一个记录的loss的回调函数
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# 构建一个自定义的TensorBoard类，专门用来记录batch中的数据变化
class BatchTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False):
        super(BatchTensorBoard, self).__init__()
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images
        self.batch = 0
        self.batch_queue = set()

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        self.batch = self.batch + 1

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = float(value)
            summary_value.tag = "batch_" + name
            if (name, self.batch) in self.batch_queue:
                continue
            self.writer.add_summary(summary, self.batch)
            self.batch_queue.add((name, self.batch))
        self.writer.flush()


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
    plt.savefig("Performance.jpg")


# paths
path = 'D:/Project/cifar/cifar10/'
model_path = path + 'models/'
log_path = path + 'logs/'
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
tensorboard = TensorBoard(log_dir='/home/tensorflow/log/softmax/epoch')
my_tensorboard = BatchTensorBoard(log_dir='/home/tensorflow/log/softmax/batch')


early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history = vgg.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(x_test, y_test), callbacks=[ckpt])
# history = vgg.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(x_test, y_test), callbacks=[tensorboard, my_tensorboard])

score = vgg.model.evaluate(x_test, y_test, verbose=1)

print("====================================")
print("====================================")
print("val_loss:{0}".format(score[0]))
print("val_acc :{0}".format(score[1]))
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
