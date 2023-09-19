# -*- coding: utf-8 -*-
from keras.models import Model, Sequential, save_model, load_model
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, \
    Dropout, BatchNormalization, ReLU, DepthwiseConv2D, Multiply, UpSampling2D, Lambda, Add, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.layers import Concatenate
from keras.utils import plot_model
from keras import regularizers
import tensorflow as tf
import imageio

from CBAM import channel_attention, spatial_attention
from DANet import PAM, CAM

def channel_split(x, name=''):
    # 输入进来的通道数
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    # 对通道数进行分割
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c

def channle_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    # 通道交换
    x = tf.reshape(x, [-1, height, width, 2, channels_per_split])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, height, width, channels])
    return x


# def channle_shuffle(inputs):
#     """Shuffle the channel
#     Args:
#         inputs: 4D Tensor
#         group: int, number of groups
#     Returns:
#         Shuffled 4D Tensor
#     """
#     group = 8
#     in_shape = inputs.get_shape().as_list()
#     h, w, in_channel = in_shape[1:]
#     assert in_channel % group == 0
#     l = tf.reshape(inputs, [-1, h, w, in_channel // group, group])
#     l = tf.transpose(l, [0, 1, 2, 4, 3])
#     l = tf.reshape(l, [-1, h, w, in_channel])
#
#     return l


def MC_Unit1(inputs):
    c_hat, c = channel_split(inputs)
    inputs = c
    stride = 1
    x = BatchNormalization()(inputs)
    c_hat = BatchNormalization()(c_hat)
    xx = x

    mc_conv = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    mc_bn = BatchNormalization()(mc_conv)
    dp1 = DepthwiseConv2D(kernel_size=(3, 3), strides=stride, depth_multiplier=1,
                          padding='same', use_bias=False)(mc_bn)
    dp2 = DepthwiseConv2D(kernel_size=(5, 5), strides=stride, depth_multiplier=1,
                          padding='same', use_bias=False)(mc_bn)
    dp3 = DepthwiseConv2D(kernel_size=(7, 7), strides=stride, depth_multiplier=1,
                          padding='same', use_bias=False)(mc_bn)
    x2 = Multiply()([dp1, dp2])
    x2 = BatchNormalization()(x2)
    x2 = Multiply()([x2, dp3])
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(64, (1, 1), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x = Concatenate()([c_hat, x2])
    # x = channle_shuffle(x, 2)
    x = Lambda(channle_shuffle)(x)
    return x


def MC_Unit2(inputs):
    c_hat, c = channel_split(inputs)
    inputs = c
    stride = 2
    x = BatchNormalization()(inputs)
    c_hat = BatchNormalization()(c_hat)
    xx = x
    if stride == 2:
        dp = DepthwiseConv2D(kernel_size=(3, 3), strides=stride, depth_multiplier=1,
                             padding='same', use_bias=False)(c_hat)
        mc_conv = Conv2D(64, (1, 1), activation='relu', padding='same')(dp)
        xx = BatchNormalization(name='mc_bn1')(mc_conv)

    mc_conv = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    mc_bn = BatchNormalization()(mc_conv)
    dp1 = DepthwiseConv2D(kernel_size=(3, 3), strides=stride, depth_multiplier=1,
                          padding='same', use_bias=False)(mc_bn)
    dp2 = DepthwiseConv2D(kernel_size=(5, 5), strides=stride, depth_multiplier=1,
                          padding='same', use_bias=False)(mc_bn)
    dp3 = DepthwiseConv2D(kernel_size=(7, 7), strides=stride, depth_multiplier=1,
                          padding='same', use_bias=False)(mc_bn)
    x2 = Multiply()([dp1, dp2])
    x2 = BatchNormalization()(x2)
    x2 = Multiply()([x2, dp3])
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(64, (1, 1), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x = Concatenate()([xx, x2])
    #x = channle_shuffle(x, 2)
    x = Lambda(channle_shuffle)(x)
    return x


def MC_Unit(inputs, stride):
    x = BatchNormalization()(inputs)
    xx = x
    if stride == 2:
        dp = DepthwiseConv2D(kernel_size=(3, 3), strides=stride, depth_multiplier=1,
                             padding='same', use_bias=False)(x)
        mc_conv = Conv2D(32, (1, 1), activation='relu', padding='same')(dp)
        xx = BatchNormalization(name='mc_bn1')(mc_conv)

    mc_conv = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    mc_bn = BatchNormalization()(mc_conv)
    dp1 = DepthwiseConv2D(kernel_size=(3, 3), strides=stride, depth_multiplier=1,
                          padding='same', use_bias=False)(mc_bn)
    dp2 = DepthwiseConv2D(kernel_size=(5, 5), strides=stride, depth_multiplier=1,
                          padding='same', use_bias=False)(mc_bn)
    dp3 = DepthwiseConv2D(kernel_size=(7, 7), strides=stride, depth_multiplier=1,
                          padding='same', use_bias=False)(mc_bn)
    x2 = Multiply()([dp1, dp2])
    x2 = Multiply()([x2, dp3])
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(32, (1, 1), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x = Concatenate()([xx, x2])
    x = channle_shuffle(x, 2)
    return x


# def GAP(inputs):
#
#

def MSAN(num_pc, img_rows, img_cols, nb_classes, falg_summary=False, pretrained_weights=None, model_plot=False):
    inputs = Input(shape=[img_rows, img_cols, num_pc], name='Input')

    conv1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
    bn = BatchNormalization(name='bn_conv')(conv1)
    pool = MaxPooling2D((3, 3), name='pool')(bn)
    ca = channel_attention(pool)
    sa = spatial_attention(pool)
    #ca = PAM()(pool)
    #sa = CAM()(pool)
    x = Concatenate()([ca, sa])
    #x = _PSA(pool)
    # x = MC_Unit(x, 1)
    # x = MC_Unit(x, 1)
    # x = MC_Unit(x, 2)
   # x = Conv2D(32, (9, 9), activation='relu', padding='same', name='gloal')(pool)
    
    xx = Lambda(MC_Unit2)(pool)
    xx = Lambda(MC_Unit1)(xx)
    xx = Lambda(MC_Unit1)(xx)
    #xx = Lambda(MC_Unit2)(xx)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=2, depth_multiplier=1,
                         padding='same', use_bias=False)(x)
    xx = Conv2D(64, (1, 1), activation='relu', padding='same')(xx)
    xx = Concatenate()([x, xx])
    #xx = Lambda(MC_Unit2)(xx)
    
    # xx = Lambda(MC_Unit1)(xx)
    # xx = Lambda(MC_Unit2)(xx)
    # xx = Lambda(MC_Unit2)(xx)
    #xx = Lambda(MC_Unit2)(xx)
    conv2 = Conv2D(64, (1, 1), activation='relu', padding='same', name='conv2')(x)
    # GAP(conv2)
    #flatten = Flatten(name='FLATTEN1')(conv2)
    pool1 = GlobalAveragePooling2D()(conv2)
    
    outputs = Dense(nb_classes, activation="softmax", kernel_regularizer=regularizers.l2(0.01))(pool1) #fc_size
    # dense = Dense(32, activation='relu', name='dense')(flatten)
    #outputs = Dense(nb_classes, activation='softmax', name='softmax')(flatten)
    # x = UpSampling2D(output_size=(img_rows, img_cols))(conv2)
    # outputs = Conv2D(1, 1, activation='softmax')(conv2)
    # x = UpSampling2D(output_size=(img_rows, img_cols))(x)
    model = Model(inputs=[inputs], outputs=[outputs])
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
   
    model.compile(optimizer =  adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if falg_summary:
        model.summary()
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    if model_plot is True:
        plot_model(model, to_file='MSAN.png')
    return model

x = MSAN(200, 27, 27, 9, falg_summary=True)
