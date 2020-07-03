# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import tensorflow as tf
from my_transformer import Transformer, create_padding_mask
from my_rnn import BiRNN_SS, BiRNN_PP, BiRNN_C2, BiRNN_C3, BiRNN_C4, BiRNN_C5
from my_cnn import CNN
from utils import clean_inputs, compute_cross_entropy_loss, compute_mse_loss

class Model(object):
    
    def __init__(self, params, name):
        
        self.params = params
        self.name = name
        
        self.transformer = Transformer(num_layers=self.params["transfomer_layers"],
                                       d_model=self.params["d_input"],
                                       num_heads=self.params["transfomer_num_heads"],
                                       rate=self.params["dropout_rate"])

        self.cnn = CNN(num_layers=self.params["cnn_layers"],
                       channels=self.params["cnn_channels"])
        
        if self.name[:2] == "ss":
            self.birnn = BiRNN_SS(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ss8_output=self.params["d_ss8_output"],
                                  ss3_output=self.params["d_ss3_output"])
            print ("use ss model...")
            
        elif self.name[:2] == "pp":
            self.birnn = BiRNN_PP(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  phipsi_output=self.params["d_phipsi_output"])
            print ("use pp model...")

        elif self.name[:2] == "c2":
            self.birnn = BiRNN_C2(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ss8_output=self.params["d_ss8_output"],
                                  ss3_output=self.params["d_ss3_output"],
                                  phipsi_output=self.params["d_phipsi_output"])
            print ("use combine ss-pp model...")

        elif self.name[:2] == "c3":
            self.birnn = BiRNN_C3(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ss8_output=self.params["d_ss8_output"],
                                  ss3_output=self.params["d_ss3_output"],
                                  phipsi_output=self.params["d_phipsi_output"],
                                  csf_output=self.params["d_csf_output"])
            print ("use combine ss-pp-csf model...")

        elif self.name[:2] == "c4":
            self.birnn = BiRNN_C4(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ss8_output=self.params["d_ss8_output"],
                                  ss3_output=self.params["d_ss3_output"],
                                  phipsi_output=self.params["d_phipsi_output"],
                                  csf_output=self.params["d_csf_output"],
                                  asa_output=self.params["d_asa_output"])
            print ("use combine ss-pp-csf-asa model...")
            
        elif self.name[:2] == "c5":
            self.birnn = BiRNN_C5(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ss8_output=self.params["d_ss8_output"],
                                  ss3_output=self.params["d_ss3_output"],
                                  phipsi_output=self.params["d_phipsi_output"],
                                  csf_output=self.params["d_csf_output"],
                                  asa_output=self.params["d_asa_output"],
                                  rota_output=self.params["d_rota_output"])
            print ("use combine ss-pp-csf-asa-rota model...")

    def inference(self, x, x_mask, y, y_mask, training):

        encoder_padding_mask = create_padding_mask(x_mask)    
        
        x = clean_inputs(x, x_mask, self.params["d_input"])
        
        transformer_out = self.transformer(x, encoder_padding_mask, training=training)
        cnn_out = self.cnn(x, training=training)
        x = tf.concat((x, cnn_out, transformer_out), -1)
        
        x = clean_inputs(x, x_mask, 3*self.params["d_input"])
        
        if self.name[:2] == "ss":
            ss8_predictions, ss3_predictions = \
                self.birnn(x, x_mask, training=training) 
            loss = None
            if training == True:
                loss = compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8]) + \
                        compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33])                      
            return ss8_predictions, ss3_predictions, loss 
        
        elif self.name[:2] == "pp":
            phipsi_predictions = \
                self.birnn(x, x_mask, training=training) 
            loss = None
            if training == True:
                loss = compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15])
            return phipsi_predictions, loss

        elif self.name[:2] == "c2":
            ss8_predictions, ss3_predictions, phipsi_predictions = \
                self.birnn(x, x_mask, training=training) 
            loss = None
            if training == True:
                loss = compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8]) + \
                        compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        4*compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15])
            return ss8_predictions, ss3_predictions, phipsi_predictions, loss

        elif self.name[:2] == "c3":
            ss8_predictions, ss3_predictions, phipsi_predictions, csf_predictions = \
                self.birnn(x, x_mask, training=training) 
            loss = None
            if training == True:
                loss = compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8]) + \
                        compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        4*compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15]) + \
                        0.1*compute_mse_loss(csf_predictions, y[:,:,8:11], y_mask[:,:,8:11])
            return ss8_predictions, ss3_predictions, phipsi_predictions, csf_predictions, loss

        elif self.name[:2] == "c4":
            ss8_predictions, ss3_predictions, phipsi_predictions, csf_predictions, asa_predictions = \
                self.birnn(x, x_mask, training=training) 
            loss = None
            if training == True:
                loss = compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8]) + \
                        compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        4*compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15]) + \
                        0.1*compute_mse_loss(csf_predictions, y[:,:,8:11], y_mask[:,:,8:11]) + \
                        3*compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1))
            return ss8_predictions, ss3_predictions, phipsi_predictions, csf_predictions, asa_predictions, loss

        elif self.name[:2] == "c5":
            ss8_predictions, ss3_predictions, phipsi_predictions, csf_predictions, asa_predictions, rota_predictions = \
                self.birnn(x, x_mask, training=training) 
            loss = None
            if training == True:
                loss = compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8]) + \
                        compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        4*compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15]) + \
                        0.1*compute_mse_loss(csf_predictions, y[:,:,8:11], y_mask[:,:,8:11]) + \
                        3*compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1)) + \
                        compute_mse_loss(rota_predictions, y[:,:,15:23], y_mask[:,:,15:23])
            return ss8_predictions, ss3_predictions, phipsi_predictions, csf_predictions, asa_predictions, rota_predictions, loss
                
    def save_model(self):
        print ("save model:", self.name)
        self.transformer.save_weights(os.path.join(self.params["save_path"], self.name + '_trans_model_weight'))
        self.cnn.save_weights(os.path.join(self.params["save_path"], self.name + '_cnn_model_weight'))
        self.birnn.save_weights(os.path.join(self.params["save_path"], self.name + '_birnn_model_weight'))

    def load_model(self):
        print ("load model:", self.name)
        self.transformer.load_weights(os.path.join(self.params["save_path"], self.name + '_trans_model_weight'))
        self.cnn.load_weights(os.path.join(self.params["save_path"], self.name + '_cnn_model_weight'))
        self.birnn.load_weights(os.path.join(self.params["save_path"], self.name + '_birnn_model_weight'))





