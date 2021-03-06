# -*- coding:utf-8 -*-
import numpy as np

from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras import initializers
import keras.backend as K
from keras.callbacks import Callback, TensorBoard
import tensorflow as tf
from time import sleep
np.random.seed(1337)  # for reproducibility

# 设置线程
THREADS_NUM = 3
tf.ConfigProto(intra_op_parallelism_threads=THREADS_NUM)

# vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))
vgg_mean = np.array([125.307, 122.950, 113.865], dtype=np.float32).reshape((3, 1, 1))

def vgg_preprocess(x):
    x = x - vgg_mean
    # x -= np.mean(x, axis=0)  # zero-center
    # x /= np.std(x, axis=0)  # normalize
    return x[:, ::-1]  # reverse axis rgb->bgr


class Vgg16BN():
    """The VGG 16 Imagenet model with Batch Normalization for the Dense Layers"""

    def __init__(self, size=(32, 32), n_classes=10, lr=0.001, batch_size=64):
        self.path = 'D:/Project/cifar/cifar10/'
        self.size = size
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        # self.build_vgg16()
        # self.build_vgg_simple()
        # self.build_simple_net()
        # self.build_net_78()
        # self.build_net_84()
        # self.build_Seenta_net()
        self.buildct_net()  # Seenta_net上添加全局pooling

        self.weights_file = self.path + 'vgg16_bn.h5'  # download from: http://www.platform.ai/models/
        self.save_model()

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    def predict(self, data):
        return self.model.predict(data)

    def ConvBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Conv2D(filters, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding="same"))

    def FCBlock(self, n_dense, p_dropout):
        model = self.model
        model.add(Dense(n_dense, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(p_dropout))

    def build_vgg16(self, ft=True):
        # model = self.model = Sequential()
        # model.add(Lambda(vgg_preprocess, input_shape=(3,) + self.size))
        model = self.model = Sequential()
        # model.add(Lambda(vgg_preprocess, input_shape=(3,) + self.size))
        model.add(Lambda(vgg_preprocess, input_shape=(3,)+self.size))

        # 这边展示的16层的VGG，还有19层的，即512个filter size的层叠加4层
        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        model.add(Flatten())
        self.FCBlock(4096, 0.5)
        self.FCBlock(4096, 0.5)
        print(model.summary())

        model.add(Dense(self.n_classes, activation='softmax'))

        # model.load_weights(self.weights_file)

        if ft:
            self.finetune()

        self.compile()

    def build_vgg_simple(self, ft=True):
        model = self.model = Sequential()
        model.add(Lambda(vgg_preprocess, input_shape=(3,) + self.size))

        self.ConvBlock(2, 32)
        self.ConvBlock(2, 64)

        model.add(Flatten())
        self.FCBlock(1024, 0.25)
        self.FCBlock(256, 0.25)
        model.add(Dense(self.n_classes, activation='softmax'))

        if ft:
            self.finetune()

        self.compile()

    def build_simple_net(self):
        model = self.model = Sequential()
        print("initial shape {0}".format((3,) + self.size))
        model.add(Lambda(vgg_preprocess, input_shape=(3,) + self.size))

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
        print(model.summary())

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
        print(model.summary())

        # model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
        print(model.summary())

        model.add(Flatten())

        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.25))

        model.add(Dense(self.n_classes, activation='softmax'))

        # if ft:
        #     self.finetune()

        self.compile()

    def build_net_84(self):
        model = self.model = Sequential()
        model.add(Lambda(vgg_preprocess, input_shape=(3,) + self.size))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
        model.add(BatchNormalization())

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
        model.add(BatchNormalization())

        print(model.summary())

        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.n_classes, activation='softmax'))

        # if ft:
        #     self.finetune()

        self.compile()

    def build_net_78(self):
        model = self.model = Sequential()
        model.add(Lambda(vgg_preprocess, input_shape=(3,) + self.size))

        model.add(Conv2D(32, (5, 5), padding='valid', activation='relu'))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (5, 5), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))

        model.add(Dense(self.n_classes, activation='softmax'))

        self.compile()

    def build_Seenta_net(self):
        """
        可用的initialization方法：random_normal(stddev=0.0001), Orthogonal(), glorot_uniform(), lecun_uniform()
        :return:
        """
        model = self.model = Sequential()
        print("initial shape {0}".format((3,) + self.size))
        dr1 = 0.2

        model.add(Lambda(vgg_preprocess, input_shape=(3,) + self.size))

        model.add(ZeroPadding2D((2, 2)))
        model.add(Conv2D(32, (5, 5), activation='relu', kernel_initializer=initializers.Orthogonal()))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(BatchNormalization())
        # model.add(Dropout(dr1))
        # print(model.summary())

        model.add(ZeroPadding2D((2, 2)))
        model.add(Conv2D(32, (5, 5), activation='relu', kernel_initializer=initializers.Orthogonal()))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(BatchNormalization())
        # model.add(Dropout(dr1))
        # print(model.summary())

        model.add(ZeroPadding2D((2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer=initializers.Orthogonal()))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(BatchNormalization())
        # model.add(Dropout(dr1))
        # print(model.summary())

        model.add(Flatten())
        model.add(Dense(64, activation='relu', kernel_initializer=initializers.Orthogonal()))
        model.add(Dropout(dr1))
        # model.add(BatchNormalization())
        model.add(Dense(self.n_classes, activation='softmax', kernel_initializer=initializers.Orthogonal()))
        print(model.summary())

        # if ft:
        #     self.finetune()

        self.compile()

    def conv2d_bn(self, model, filters, num_row, num_col, initialization, padding='same', strides=(1, 1)):
        """Utility function to apply conv + BN.
        Returns: Output tensor after applying `Conv2D` and `BatchNormalization`.
        """
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3
        model.add(Conv2D(filters, (num_row, num_col), kernel_initializer=initialization, strides=strides, padding=padding, use_bias=False))
        model.add(BatchNormalization(axis=bn_axis, scale=False))
        model.add(Activation('relu'))
        return model

    def build_ct_net(self):
        """
        可用的initialization方法：random_normal(stddev=0.0001), Orthogonal(), glorot_uniform(), lecun_uniform()
        :return:
        """
        model = self.model = Sequential()
        print("initial shape {0}".format((3,) + self.size))
        dr1 = 0.2
        initial_dict = {'orthogonal': initializers.Orthogonal(), 'he_n': "he_normal"}

        model.add(Lambda(vgg_preprocess, input_shape=(3,) + self.size))

        model = self.conv2d_bn(model, 32, 3, 3, initial_dict['orthogonal'])
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model = self.conv2d_bn(model, 32, 3, 3, initial_dict['orthogonal'])
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        model = self.conv2d_bn(model, 64, 3, 3, initial_dict['orthogonal'])
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
        # print(model.summary())

        model.add(Flatten())
        model.add(Dense(64, activation='relu', kernel_initializer=initializers.Orthogonal()))
        model.add(Dropout(dr1))
        # model.add(BatchNormalization())
        model.add(Dense(self.n_classes, activation='softmax', kernel_initializer=initializers.Orthogonal()))
        print(model.summary())

        # if ft:
        #     self.finetune()

        self.compile()

    def finetune(self):
        model = self.model
        model.pop()
        for layer in model.layers:
            layer.trainable=False
        model.add(Dense(self.n_classes, activation='softmax'))

    def compile(self):
        self.model.compile(optimizer=Adam(lr=self.lr), loss='categorical_crossentropy', metrics=['accuracy'])

    def test(self, test_path, nb_test_samples, aug=False):
        if aug:
            test_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                               channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                               horizontal_flip=True)
        else:
            test_datagen = ImageDataGenerator()

        test_gen = test_datagen.flow_from_directory(test_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode=None, shuffle=False)

        return self.model.predict_generator(test_gen, val_samples=nb_test_samples), test_gen.filenames

    def save_model(self):
        model = self.model
        model.save(self.weights_file)

# def fit(self, trn_path, val_path, nb_trn_samples, nb_val_samples, nb_epoch=1, callbacks=None, aug=False):
    #     if aug:
    #         train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
    #                                            channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
    #                                            horizontal_flip=True)
    #     else:
    #         train_datagen = ImageDataGenerator()
    #
    #     trn_gen = train_datagen.flow_from_directory(trn_path, target_size=self.size, batch_size=self.batch_size,
    #                                                   class_mode='categorical', shuffle=True)
    #
    #     val_gen = ImageDataGenerator().flow_from_directory(val_path, target_size=self.size, batch_size=self.batch_size,
    #                                                        class_mode='categorical', shuffle=True)
    #
    #     self.model.fit_generator(trn_gen, samples_per_epoch=nb_trn_samples, nb_epoch=nb_epoch, verbose=2,
    #             validation_data=val_gen, nb_val_samples=nb_val_samples, callbacks=callbacks)
