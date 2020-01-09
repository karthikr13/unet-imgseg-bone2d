# !/usr/bin/python3

### Description    : Model training and evaluation; predictions.
### Version        : Python = 3.7; Tensorflow = 2.1.0.
### Author         : Haoran Xi.
### Created        : 2020/01/08
### Last updated   : 2020/01/08

from keras.models import Model
from keras import losses
from keras.layers import Input, merge, Lambda, Softmax, Convolution2D, Convolution3D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

def get_unet(loss, metric_list):
    K.clear_session()
    inputs =Input((966,1296,3))
    conv0 = Convolution2D(1, 1, activation='relu', padding='same')(inputs)

    conv1 = Convolution2D(32, 3, activation='relu', padding='same')(conv0)
    conv1 = Convolution2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Convolution2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Convolution2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = Convolution2D(256, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, activation='relu', padding='same')(pool4)
    conv5 = Convolution2D(512, 3, activation='relu', padding='same')(conv5)

    up1 = UpSampling2D(size=(2, 2))(conv5)
    up1 = Convolution2D(256, 2, activation='relu', padding='same')(up1)
    up1 = merge.concatenate([up1, conv4])  # 120, 162, 512
    conv6 = Convolution2D(256, 3, activation='relu', padding='same')(up1)
    conv6 = Convolution2D(256, 3, activation='relu', padding='same')(conv6)
    conv6 = ZeroPadding2D((1, 0))(conv6)

    up2 = UpSampling2D(size=(2, 2))(conv6)
    up2 = Convolution2D(128, 2, activation='relu', padding='same')(up2)
    up2 = merge.concatenate([Lambda(lambda x: x[:, 1:242])(up2), conv3])
    conv7 = Convolution2D(128, 3, activation='relu', padding='same')(up2)
    conv7 = Convolution2D(128, 3, activation='relu', padding='same')(conv7)
    conv7 = ZeroPadding2D((1, 0))(conv7)

    up3 = UpSampling2D(size=(2, 2))(conv7)
    up3 = Convolution2D(64, 2, activation='relu', padding='same')(up3)
    up3 = merge.concatenate([Lambda(lambda x: x[:, 1:484])(up3), conv2])
    conv8 = Convolution2D(64, 3, activation='relu', padding='same')(up3)
    conv8 = Convolution2D(64, 3, activation='relu', padding='same')(conv8)

    up4 = UpSampling2D(size=(2, 2))(conv8)
    up4 = Convolution2D(32, 2, activation='relu', padding='same')(up4)
    up4 = merge.concatenate([up4, conv1])  # 966, 1296, 64
    conv9 = Convolution2D(32, 3, activation='relu', padding='same')(up4)
    conv9 = Convolution2D(32, 3, activation='relu', padding='same')(conv9)

    conv10 = Convolution2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-5), loss=loss, metrics=metric_list)

    return model