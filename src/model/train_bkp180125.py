# -*- coding:utf-8 -*-
import os
import numpy as np

from glob import glob
from shutil import copyfile
from src.model.vgg_bn import Vgg16BN
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10

PATH = 'D:\Project\cifar\cifar10'
DATA_PATH = ROOT_DIR + '/data'

# paths
data_path = DATA_HOME_DIR + '/'
train_path = data_path + '/train/'
valid_path = data_path + '/valid/'
test_path = DATA_HOME_DIR + '/test/'
model_path = ROOT_DIR + '/models/'
submission_path = ROOT_DIR + '/submissions/'

# data
img_width, img_height = 224, 224
batch_size = 64
nb_train_samples = 23000
nb_valid_samples = 2000
nb_test_samples = 12500
classes = ["cats", "dogs"]
n_classes = len(classes)

# model
nb_epoch = 10
nb_aug = 5
lr = 0.001

vgg = Vgg16BN(size=(img_width, img_height), n_classes=n_classes, batch_size=batch_size, lr=lr)
model = vgg.model

model.summary()

info_string = "{0}x{1}_{2}epoch_{3}aug_{4}lr_vgg16-bn".format(img_width, img_height, nb_epoch, nb_aug, lr)
ckpt_fn = model_path + '{val_loss:.2f}-loss_' + info_string + '.h5'

ckpt = ModelCheckpoint(filepath=ckpt_fn,
                      monitor='val_loss',
                      save_best_only=True,
                      save_weights_only=True)

vgg.fit(train_path, valid_path,
          nb_trn_samples=nb_train_samples,
          nb_val_samples=nb_valid_samples,
          nb_epoch=nb_epoch,
          callbacks=[ckpt],
          aug=nb_aug)

# generate predictions
for aug in range(nb_aug):
    print("Generating predictions for Augmentation {0}...",format(aug+1))
    if aug == 0:
        predictions, filenames = vgg.test(test_path, nb_test_samples, aug=nb_aug)
    else:
        aug_pred, filenames = vgg.test(test_path, nb_test_samples, aug=nb_aug)
        predictions += aug_pred

print("Averaging Predictions Across Augmentations...")
predictions /= nb_aug

# clip predictions
c = 0.01
preds = np.clip(predictions, c, 1-c)

sub_file = submission_path + info_string + '.csv'

with open(sub_file, 'w') as f:
    print("Writing Predictions to CSV...")
    f.write('id,label\n')
    for i, image_name in enumerate(filenames):
        pred = ['%.6f' % p for p in preds[i, :]]
        if i % 2500 == 0:
            print(i, '/', nb_test_samples)
        f.write('%s,%s\n' % (os.path.basename(image_name).replace('.jpg', ''), (pred[1])))
    print("Done.")