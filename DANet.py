import tensorflow as tf
from keras.layers import *
def PAM_Module(x):
    gamma = tf.Variable(tf.ones(1))
    x_origin = x
    batch_size, H, W, Channel = x.shape
    proj_query = Conv2D(kernel_size=1, filters=Channel // 8, padding='same')(x)
    proj_key = Conv2D(kernel_size=1, filters=Channel // 8, padding='same')(x)
    proj_value = Conv2D(kernel_size=1, filters=Channel, padding='same')(x)
    proj_query, proj_key, proj_value = tf.transpose(proj_query, [0, 3, 1, 2]), tf.transpose(proj_key, [0, 3, 1, 2]), tf.transpose(proj_value, [0, 3, 1, 2])
    proj_key = tf.reshape(proj_key, (-1, Channel//8, H*W))
    proj_query = tf.transpose(tf.reshape(proj_query, (-1, Channel//8, H*W)), [0, 2, 1])
    energy = tf.matmul(proj_query, proj_key)
    attention = tf.nn.softmax(energy)
    proj_value = tf.reshape(proj_value, (-1, Channel, H*W))
    out = tf.matmul(proj_value, tf.transpose(attention, [0, 2, 1]))
    out = tf.reshape(out, (-1, Channel, H, W))
    out = tf.transpose(out, [0, 2, 3, 1])
    out = add([out*gamma, x_origin])

    return out
def CAM_Module(x):
    """
    通道注意力 Channel Attention Moudle
    :param x: 输入数组[B, H, W, C]
    :return: 输出数组[B, H, W, C]
    """
    gamma = tf.Variable(tf.ones(1))
    x_origin = x
    batch_size, H, W, Channel = x.shape
    x = tf.transpose(x, [0, 3, 1, 2])
    proj_query = tf.reshape(x, (-1, Channel, H*W))
    proj_key = tf.transpose(proj_query, [0, 2, 1])
    energy = tf.matmul(proj_query, proj_key)
    energy_new = tf.reduce_max(energy, axis=-1, keepdims=True)
    energy_new = tf.repeat(energy_new, Channel, axis=-1)
    energy_new = energy_new - energy
    attention = tf.nn.softmax(energy_new)
    proj_value = proj_query
    out = tf.matmul(attention, proj_value)
    out = tf.reshape(out, (-1, Channel, H, W))
    out = tf.transpose(out, [0, 2, 3, 1])
    out = add([out*gamma, x_origin])

    return out
