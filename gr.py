#%%

import cv2
import urllib
import numpy as np
import tensorflow as tf
from urllib.request import urlopen
from matplotlib import pyplot as plt

#%%
def vid_input:
    url = 'http://192.168.43.1:8080/shot.jpg'

    while True:
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()),dtype=np.uint8)
        img_clr = cv2.imdecode(img_np,-1)
        img_gray = cv2.cvtColor(img_clr, cv2.COLOR_BGR2GRAY)

        v = np.median(img_gray)
        sigma = 0.6

        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))

        filter = cv2.Canny(img_gray,lower,upper)
    
        cv2.imshow('Original',img_clr)
        cv2.imshow('Canny Filter',filter)

        if ord('q')==cv2.waitKey(10):
            exit(0)
        
#%%

img_rows = 200 
img_cols = 200
img_channels = 1
batch_size = 32
nb_classes = 5
nb_epoch = 15
nb_filters = 32
pool_shape = 2
filter_shape = 3
output = ["OK", "NOTHING","PEACE", "PUNCH", "STOP"]

#%%

x = tf.placeholder(tf.float32, [None, img_rows*img_cols])
x_shaped = tf.reshape(x, [1, img_rows, img_cols, 1])
y = tf.placeholder(tf.float32, [None, 10])

#%%

def conv_layer(input_data, img_channels, nb_filters, filter_shape, pool_shape, name):
    conv_filter_shape = [filter_shape, filter_shape, img_channels, nb_filters]
    weights = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev = 0.03), name = name+'_W')
    bias = tf.Variable(tf.truncated_normal([nb_filters]), name=name+'_b')
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
    out_layer += bias
    out_layer = tf.nn.relu(out_layer)
    ksize = [1, pool_shape, pool_shape, 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
    return out_layer

#%%

layer1 = conv_layer(x_shaped, img_channels, nb_filters, filter_shape, pool_shape, name = 'layer1')
layer2 = conv_layer(layer1, 32, nb_filters, filter_shape, pool_shape, name = 'layer2')

#%%

flattened = tf.reshape(layer2, [-1, 80000])

wd1 = tf.Variable(tf.truncated_normal([80000, 128], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([128], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

wd2 = tf.Variable(tf.truncated_normal([128, 5], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([5], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)

#%%

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = dense_layer2, labels = y))





















