import os
from keras.applications import MobileNet
from keras.layers import Conv2DTranspose,Conv2D,Add
from keras import Model
from keras.models import load_model

def mobilenet_8s(train_encoder = True, final_layer_activation='sigmoid',prep=True):
    '''
        This script creates a model object and loads pretrained weights 
    '''

    net = MobileNet(include_top=False, weights=None)
    if prep == True:
        net.load_weights(os.path.join('.','mn_classification_weights.h5'), by_name=True)
    else:
        net.load_weights(os.path.join('.','test_preprocessing_weights.h5'), by_name=True)

    for layer in net.layers:
        layer.trainable = train_encoder
    
    #build decoder
        predict = Conv2D(filters=1,kernel_size=1,strides=1)(net.output)
        deconv2 = Conv2DTranspose(filters=1,kernel_size=4,strides=2, padding='same', use_bias=False)(predict)
        pred_conv_pw_11_relu = Conv2D(filters=1,kernel_size=1,strides=1)(net.get_layer('conv_pw_11_relu').output)
        fuse1 = Add()([deconv2, pred_conv_pw_11_relu])
        pred_conv_pw_5_relu = Conv2D(filters=1,kernel_size=1,strides=1)(net.get_layer('conv_pw_5_relu').output)
        deconv2fuse1 = Conv2DTranspose(filters=1,kernel_size=4,strides=2, padding='same', use_bias=False)(fuse1)
        fuse2 = Add()([deconv2fuse1, pred_conv_pw_5_relu])
        deconv8 = Conv2DTranspose(filters=1,kernel_size=16,strides=8, padding='same', use_bias=False, activation=final_layer_activation)(fuse2)

        return Model(inputs=net.input,outputs=deconv8)


def mobilenet_16s(train_encoder = True, final_layer_activation='sigmoid',prep=True):
    '''
        This script creates a model object and loads pretrained weights 
    '''
    net = MobileNet(include_top=False, weights=None)
    if prep == True:
        net.load_weights(os.path.join('.','mn_classification_weights.h5'), by_name=True)
    else:
        net.load_weights(os.path.join('.','test_preprocessing_weights.h5'), by_name=True)

    for layer in net.layers:
        layer.trainable = train_encoder
    
    #build decoder
    predict = Conv2D(filters=1,kernel_size=1,strides=1)(net.output)
    deconv2 = Conv2DTranspose(filters=1,kernel_size=4,strides=2, padding='same', use_bias=False)(predict)
    pred_conv_pw_11_relu = Conv2D(filters=1,kernel_size=1,strides=1)(net.get_layer('conv_pw_11_relu').output)
    fuse1 = Add()([deconv2, pred_conv_pw_11_relu])
    deconv16 = Conv2DTranspose(filters=1,kernel_size=32,strides=16, padding='same', use_bias=False, activation=final_layer_activation)(fuse1)

    return Model(inputs=net.input,outputs=deconv16)


def mobilenet_32s(train_encoder = True, final_layer_activation='sigmoid',prep=True):
    '''
        This script creates a model object and loads pretrained weights 
    '''

    net = MobileNet(include_top=False, weights=None)
    if prep == True:
        net.load_weights(os.path.join('.','mn_classification_weights.h5'), by_name=True)
    else:
        net.load_weights(os.path.join('.','test_preprocessing_weights.h5'), by_name=True)

    for layer in net.layers:
        layer.trainable = train_encoder
    
    #build decoder
    predict = Conv2D(filters=1,kernel_size=1,strides=1)(net.output)
    deconv32 = Conv2DTranspose(filters=1,kernel_size=64,strides=32, padding='same', use_bias=False, activation=final_layer_activation)(predict)

    return Model(inputs=net.input,outputs=deconv32)