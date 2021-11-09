import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook, tnrange, tqdm
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# Set some parameters
im_width = 128
im_height = 128
border = 5
path_train = './resource/input/train/'
path_test = './resource/input/test/'

# 加载图片
# Get and resize train images and masks
def get_data(path, train=True):
    ids = next(os.walk(path + "images"))[2]
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(path + '/images/' + id_, color_mode="grayscale")
        x_img = img_to_array(img)
        x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True)

        # Load masks
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_, color_mode="grayscale"))
            mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X
# 拆分训练集和验证集
def split(X,y):
    global X_train, X_valid, y_train, y_valid
    a,b,c,d = train_test_split(X, y, test_size=0.15, random_state=2019)
    X_train, X_valid, y_train, y_valid = a, b, c, d
    print("X_train:\n",X_train)
    # print(X_valid)
    print("y_train:\n",y_train)
    # print(y_valid)

# 随机选一张训练图片原图+mask进行展示
def plot(X_train,y_train):
    # Check if training data looks all right

    ix = random.randint(0, len(X_train))
    has_mask = y_train[ix].max() > 0
    # print("has_mask：", has_mask)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0].imshow(X_train[ix, ..., 0], cmap='seismic', interpolation='bilinear')
    if has_mask:
        ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Seismic')

    ax[1].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')
    ax[1].set_title('Salt');
    # 不加plt.show()，imshow不显示图像
    plt.show()

# 构建U-NET网络模型
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch] > 0, B[batch] > 0
        #         if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
        #             metric.append(0)
        #             continue
        #         if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
        #             metric.append(0)
        #             continue
        #         if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
        #             metric.append(1)
        #             continue

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)

# 数据增强
def data_gen():
    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 2019
    bs = 32

    image_generator = image_datagen.flow(X_train, seed=seed, batch_size=bs, shuffle=True)
    mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=bs, shuffle=True)

    # Just zip the two generators to get a generator that provides augmented images and masks at the same time
    train_generator = zip(image_generator, mask_generator)

def plot_iou(results):
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r",
             label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend();

def callbacks():
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.000001, verbose=1),
        ModelCheckpoint('./model-tgs-salts.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]
    return callbacks

def show_flipped_images(x):
    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(x[:,:,0], cmap='seismic')
    ax[0].set_title('original')
    ax[1].imshow(np.fliplr(x[:,:,0]), cmap='seismic')
    ax[2].imshow(np.flipud(x[:,:,0]), cmap='seismic')
    ax[3].imshow(np.fliplr(np.flipud(x[:,:,0])), cmap='seismic')
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X, y = get_data(path_train, train=True)
    split(X,y)
    plot(X_train, y_train)

    input_img = Input((im_height, im_width, 1), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

    #model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[my_iou_metric])
    model.summary()
    data_gen()
    results = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=callbacks,validation_data=(X_valid, y_valid))
    plot_iou(results)

    # model.load_weights('./model-tgs-salt.h5')
    # # Evaluate on validation set (this must be equals to the best log_loss)
    # model.evaluate(X_valid, y_valid, verbose=1)
    # # Predict on train, and val
    # preds_train = model.predict(X_train, verbose=1)
    # preds_val = model.predict(X_valid, verbose=1)
    #
    # show_flipped_images(X_train[14])


