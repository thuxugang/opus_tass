# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Bidirectional


class BiRNN_SS(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ss8_output, ss3_output):
        super(BiRNN_SS, self).__init__()
        
        self.num_layers = num_layers
        
        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]
        
        self.dropout = keras.layers.Dropout(rate)
        
        self.ss8_layer = keras.layers.Dense(ss8_output)
        self.ss3_layer = keras.layers.Dense(ss3_output)
    
    def call(self, x, x_mask, training):
        
        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))
        
        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)
        
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        
        # predictions.shape: (batch_size, input_seq_len, output_size)
        ss8_predictions = self.ss8_layer(x)
        ss3_predictions = self.ss3_layer(x)

        return ss8_predictions, ss3_predictions

class BiRNN_PP(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 phipsi_output):
        super(BiRNN_PP, self).__init__()
        
        self.num_layers = num_layers
        
        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]
        
        self.dropout = keras.layers.Dropout(rate)
        
        self.phipsi_layer = keras.layers.Dense(phipsi_output) 
    
    def call(self, x, x_mask, training):
        
        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))
        
        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)
        
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        
        # predictions.shape: (batch_size, input_seq_len, output_size)
        phipsi_predictions = self.phipsi_layer(x)
        
        return phipsi_predictions

class BiRNN_C2(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ss8_output, ss3_output, phipsi_output):
        super(BiRNN_C2, self).__init__()
        
        self.num_layers = num_layers
        
        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]
        
        self.dropout = keras.layers.Dropout(rate)
        
        self.ss8_layer = keras.layers.Dense(ss8_output)
        self.ss3_layer = keras.layers.Dense(ss3_output)
        self.phipsi_layer = keras.layers.Dense(phipsi_output) 
    
    def call(self, x, x_mask, training):
        
        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))
        
        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)
        
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        
        # predictions.shape: (batch_size, input_seq_len, output_size)
        ss8_predictions = self.ss8_layer(x)
        ss3_predictions = self.ss3_layer(x)
        phipsi_predictions = self.phipsi_layer(x)

        return ss8_predictions, ss3_predictions, phipsi_predictions
    
class BiRNN_C3(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ss8_output, ss3_output, phipsi_output, csf_output):
        super(BiRNN_C3, self).__init__()
        
        self.num_layers = num_layers
        
        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]
        
        self.dropout = keras.layers.Dropout(rate)
        
        self.ss8_layer = keras.layers.Dense(ss8_output)
        self.ss3_layer = keras.layers.Dense(ss3_output)
        self.phipsi_layer = keras.layers.Dense(phipsi_output) 
        self.csf_layer = keras.layers.Dense(csf_output)
    
    def call(self, x, x_mask, training):
        
        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))
        
        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)
        
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        
        # predictions.shape: (batch_size, input_seq_len, output_size)
        ss8_predictions = self.ss8_layer(x)
        ss3_predictions = self.ss3_layer(x)
        phipsi_predictions = self.phipsi_layer(x)
        csf_predictions = self.csf_layer(x)

        return ss8_predictions, ss3_predictions, phipsi_predictions, csf_predictions

class BiRNN_C4(keras.Model):
    def __init__(self, num_layers, units, rate, 
                 ss8_output, ss3_output, phipsi_output, csf_output, asa_output):
        super(BiRNN_C4, self).__init__()
        
        self.num_layers = num_layers
        
        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]
        
        self.dropout = keras.layers.Dropout(rate)
        
        self.ss8_layer = keras.layers.Dense(ss8_output)
        self.ss3_layer = keras.layers.Dense(ss3_output)
        self.phipsi_layer = keras.layers.Dense(phipsi_output) 
        self.csf_layer = keras.layers.Dense(csf_output)
        self.asa_layer = keras.layers.Dense(asa_output)
        
    def call(self, x, x_mask, training):
        
        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))
        
        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)
        
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        
        # predictions.shape: (batch_size, input_seq_len, output_size)
        ss8_predictions = self.ss8_layer(x)
        ss3_predictions = self.ss3_layer(x)
        phipsi_predictions = self.phipsi_layer(x)
        csf_predictions = self.csf_layer(x)
        asa_predictions = self.asa_layer(x)

        return ss8_predictions, ss3_predictions, phipsi_predictions, csf_predictions, asa_predictions


class BiRNN_C5(keras.Model):
    def __init__(self, num_layers, units, rate, 
                 ss8_output, ss3_output, phipsi_output, csf_output, asa_output, rota_output):
        super(BiRNN_C5, self).__init__()
        
        self.num_layers = num_layers
        
        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]
        
        self.dropout = keras.layers.Dropout(rate)
        
        self.ss8_layer = keras.layers.Dense(ss8_output)
        self.ss3_layer = keras.layers.Dense(ss3_output)
        self.phipsi_layer = keras.layers.Dense(phipsi_output) 
        self.csf_layer = keras.layers.Dense(csf_output)
        self.asa_layer = keras.layers.Dense(asa_output)
        self.rota_layer = keras.layers.Dense(rota_output)
        
    def call(self, x, x_mask, training):
        
        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))
        
        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)
        
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        
        # predictions.shape: (batch_size, input_seq_len, output_size)
        ss8_predictions = self.ss8_layer(x)
        ss3_predictions = self.ss3_layer(x)
        phipsi_predictions = self.phipsi_layer(x)
        csf_predictions = self.csf_layer(x)
        asa_predictions = self.asa_layer(x)
        rota_predictions = self.rota_layer(x)

        return ss8_predictions, ss3_predictions, phipsi_predictions, csf_predictions, asa_predictions, rota_predictions
