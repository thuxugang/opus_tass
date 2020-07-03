# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import tensorflow as tf
from tensorflow import keras


class CNN(keras.Model):
    def __init__(self, num_layers, channels=8):
        
        super(CNN, self).__init__()
        
        self.num_layers = num_layers
        
        self.cnn_layers = [
            [keras.layers.Conv2D(channels, kernel_size=(11,1), padding='SAME'),
            keras.layers.Conv2D(channels, kernel_size=(21,1), padding='SAME'),
            keras.layers.Conv2D(channels, kernel_size=(31,1), padding='SAME'),
            keras.layers.Conv2D(channels, kernel_size=(41,1), padding='SAME'),
            keras.layers.Conv2D(channels, kernel_size=(51,1), padding='SAME')]
        for _ in range(self.num_layers)]

        self.bn_layers = [
            keras.layers.BatchNormalization()
        for _ in range(self.num_layers)]
        
    def call(self, x, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        x = tf.expand_dims(x, -1)
        # x.shape (batch_size, max_seq_length, embeded_size, 1)
        
        for i in range(self.num_layers):
            # x.shape (batch_size, max_seq_length, embeded_size, channels)
            x = tf.concat((self.cnn_layers[i][0](x), self.cnn_layers[i][1](x),
                            self.cnn_layers[i][2](x), self.cnn_layers[i][3](x),
                            self.cnn_layers[i][4](x)), -1)
            x = self.bn_layers[i](x, training=training)
            x = tf.nn.relu(x)

        cnn_output = tf.reduce_mean(x, axis=-1)
        
        return cnn_output
