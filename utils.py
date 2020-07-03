# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

def read_filenames(data_list):
    
    filenames = []
    f = open(data_list, 'r')
    for i in f.readlines():
        if i.strip() != "":
            filenames.append(i.strip())
    f.close()

    return filenames

ratio = 0.25
def get_enhancement(inputs, index=None):
    
    if index != None:
        return inputs[index[0]:index[1]]
    else:
        length = inputs.shape[0]
        # about half
        if np.random.randint(0,2) == 0:
            return inputs, [0, length]
        else:
            start = np.random.randint(0, int(length*ratio))
            end = length - np.random.randint(0, int(length*ratio))       
            return inputs[start:end], [start, end]
    
def read_inputs(filenames, inputs_files_path, data_enhance, input_norm):
    """
    20pssm + 30hhm + 7pc + 19psp
    """
    inputs_nopadding = []
    max_len = 0
    inputs_total_len = 0
    indices = []
    for filename in filenames:
        inputs_ = np.loadtxt((os.path.join(inputs_files_path, filename + ".inputs")))
        
        if data_enhance:
            inputs_, index = get_enhancement(inputs_)
            indices.append(index)
            
        inputs_total_len += inputs_.shape[0]
        if inputs_.shape[0] > max_len:
            max_len = inputs_.shape[0]
        inputs_nopadding.append(inputs_)
    
    inputs_padding = np.zeros(shape=(len(filenames), max_len, 76))
    inputs_mask_padding = np.ones(shape=(len(filenames), max_len))

    for i in range(len(filenames)):
        inputs_padding[i,:inputs_nopadding[i].shape[0]] = inputs_nopadding[i]
        inputs_mask_padding[i,:inputs_nopadding[i].shape[0]] = 0
        
    if input_norm:
        #(hhm - 5000) / 1000
        inputs_padding[:,:,20:50] = (inputs_padding[:,:,20:50] - 5000)/1000
        
    return inputs_padding, inputs_mask_padding, inputs_total_len, indices

def read_labels(filenames, labels_files_path, data_enhance, indices):
    """
    8ss(one-hot) + 3csf(double) + [2*(phi+psi) + 2*(x1+x2+x3+x4)](sin,cos) + asa
    8 + 3 + 4 + 8 + 1
    
    ss_labels = labels[:,:,:8]
    csf_labels = labels[:,:,8:11]
    phipsi_labels = labels[:,:,11:15]
    dihedrals_labels = labels[:,:,15:23]
    asa_labels = labels[:,:,23]
    real_phipsidihedrals=labels[:,24:30]
    """
    labels_nopadding = []
    masks_nopadding = []
    max_len = 0
    labels_total_len = 0
    for idx, filename in enumerate(filenames):
        labels_ = np.loadtxt((os.path.join(labels_files_path, filename + ".labels")))
        masks_ = np.loadtxt((os.path.join(labels_files_path, filename + ".labels_mask")))
        
        #last three csf
        masks_[-3:,8:11] = 1
        
        #make ss_3
        labels_ss3 = np.zeros((labels_.shape[0], 3))
        labels_ss3[:,0] = np.sum(labels_[:,:3],-1)
        labels_ss3[:,1] = np.sum(labels_[:,3:6],-1)
        labels_ss3[:,2] = np.sum(labels_[:,6:8],-1)
        labels_ = np.concatenate((labels_, labels_ss3),-1)
        masks_ = np.concatenate((masks_, np.zeros((masks_.shape[0], 3))),-1)
        
        #rASA
        labels_[:,23] /= 100
        
        if data_enhance:
            labels_ = get_enhancement(labels_, indices[idx])
            masks_ = get_enhancement(masks_, indices[idx])
            
        assert labels_.shape[0] == masks_.shape[0]
        labels_total_len += labels_.shape[0]
        if labels_.shape[0] > max_len:
            max_len = labels_.shape[0]
        labels_nopadding.append(labels_)
        masks_nopadding.append(masks_)
        
    labels_padding = np.zeros(shape=(len(filenames), max_len, 33))
    masks_padding = np.ones(shape=(len(filenames), max_len, 33))

    for i in range(len(filenames)):
        labels_padding[i,:labels_nopadding[i].shape[0]] = labels_nopadding[i]
        masks_padding[i,:masks_nopadding[i].shape[0]] = masks_nopadding[i]
        
    return labels_padding, masks_padding, labels_total_len

class InputReader(object):

    def __init__(self, data_list, inputs_files_path, labels_files_path, \
                 num_batch_size, input_norm=False, shuffle=False, data_enhance=False):

        self.filenames = read_filenames(data_list)
        
        self.inputs_files_path = inputs_files_path
        self.labels_files_path = labels_files_path
        self.input_norm = input_norm
        self.data_enhance = data_enhance
        
        if self.data_enhance:
            print ("use data enhancement...")
            
        if shuffle:
            self.dataset = tf.data.Dataset.from_tensor_slices(self.filenames) \
                .shuffle(len(self.filenames)).batch(num_batch_size)
        else:
             self.dataset = tf.data.Dataset.from_tensor_slices(self.filenames) \
                .batch(num_batch_size)          
        
        print ("Data Size:", len(self.filenames)) 
    
    def read_file_from_disk(self, filenames_batch):
        
        filenames_batch = [bytes.decode(i) for i in filenames_batch.numpy()]
        inputs_batch, inputs_masks_batch, inputs_total_len, indices = \
            read_inputs(filenames_batch, self.inputs_files_path, self.data_enhance, self.input_norm)
        labels_batch, labels_masks_batch, labels_total_len = \
            read_labels(filenames_batch, self.labels_files_path, self.data_enhance, indices) 
        
        inputs_batch = tf.convert_to_tensor(inputs_batch, dtype=tf.float32)
        inputs_masks_batch= tf.convert_to_tensor(inputs_masks_batch, dtype=tf.float32)
        labels_batch = tf.convert_to_tensor(labels_batch, dtype=tf.float32)
        labels_masks_batch= tf.convert_to_tensor(labels_masks_batch, dtype=tf.float32)
        
        return filenames_batch, inputs_batch, inputs_masks_batch, \
            labels_batch, labels_masks_batch, inputs_total_len, labels_total_len
    
cross_entropy_loss_func = keras.losses.CategoricalCrossentropy(
    reduction = keras.losses.Reduction.NONE, from_logits=True)

def compute_cross_entropy_loss(predictions, labels, labels_mask):
    
    # labels.shape: batch, seq_len, 8
    # labels_mask.shape: batch, seq_len, 8
    # predictions.shape: batch, seq_len, 8

    labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1], tf.shape(labels)[2]))
    labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1], tf.shape(labels_mask)[2]))
    predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1], tf.shape(predictions)[2]))

    # labels_mask.shape: batch, seq_len
    labels_mask = labels_mask[:,0]
    indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
    labels_ = tf.gather(labels, indices)
    predictions_ = tf.gather(predictions, indices)

    # loss_.shape: batch*seq_len, 8
    loss_ = cross_entropy_loss_func(labels_, predictions_)
    
    return tf.reduce_mean(loss_)

mse_loss_func = keras.losses.MeanSquaredError()

def compute_mse_loss(predictions, labels, labels_mask):
    
    # labels.shape: batch, seq_len, 4
    # labels_mask.shape: batch, seq_len, 4
    # predictions.shape: batch, seq_len, 4

    labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1]*tf.shape(labels)[2],))
    labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1]*tf.shape(labels_mask)[2],))
    predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1]*tf.shape(predictions)[2],))

    # labels_mask.shape: batch, seq_len
    indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
    labels_ = tf.gather(labels, indices)
    predictions_ = tf.gather(predictions, indices)

    # loss_.shape: batch*seq_len*3
    loss_ = mse_loss_func(labels_, predictions_)
    
    return loss_  

def cal_accurarcy(name, predictions, labels, labels_mask, total_len):
    
    if name == "SS8":
        
        labels = labels[:,:,:8]
        labels_mask = labels_mask[:,:,:8]
            
        labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1], tf.shape(labels)[2]))
        labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1], tf.shape(labels_mask)[2]))
        predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1], tf.shape(predictions)[2]))
        
        # labels_mask.shape: batch, seq_len
        labels_mask = labels_mask[:,0]
        indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
        labels_ = tf.gather(labels, indices)
        predictions_ = tf.gather(predictions, indices)
        assert total_len == labels_.shape[0] == predictions_.shape[0]
        
        accuracy = tf.cast(tf.equal(tf.argmax(labels_,1), tf.argmax(predictions_,1)), tf.float32)
    
        return accuracy.numpy()

    elif name == "SS3":
        
        labels = labels[:,:,30:33]
        labels_mask = labels_mask[:,:,30:33]
            
        labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1], tf.shape(labels)[2]))
        labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1], tf.shape(labels_mask)[2]))
        predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1], tf.shape(predictions)[2]))
        
        # labels_mask.shape: batch, seq_len
        labels_mask = labels_mask[:,0]
        indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
        labels_ = tf.gather(labels, indices)
        predictions_ = tf.gather(predictions, indices)
        assert total_len == labels_.shape[0] == predictions_.shape[0]
        
        accuracy = tf.cast(tf.equal(tf.argmax(labels_,1), tf.argmax(predictions_,1)), tf.float32)
    
        return accuracy.numpy()

    elif name == "PhiPsi":
        
        labels = labels[:,:,24:26]
        labels_mask = labels_mask[:,:,24:26]
        
        # labels.shape: batch, seq_len, 2
        # predictions.shape: batch, seq_len, 4
        labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1], 2))
        labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1], 2))
        predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1], 4))
        
        # labels.shape: batch*seq_len, 2
        # predictions.shape: batch*seq_len, 4
        labels_mask = labels_mask[:,0]
        indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
        labels_ = tf.gather(labels, indices)
        predictions = tf.gather(predictions, indices)
        assert total_len == labels_.shape[0] == predictions.shape[0]
        
        labels_ = labels_.numpy()
        predictions = predictions.numpy()

        # predictions.shape: batch*seq_len, 2
        predictions_ = np.zeros((np.shape(predictions)[0], 2))
        predictions_[:,0] = np.rad2deg(
            np.arctan2(predictions[:,0], predictions[:,1]))
        predictions_[:,1] = np.rad2deg(
            np.arctan2(predictions[:,2], predictions[:,3]))
        
        phi_diff = labels_[:,0] - predictions_[:,0]
        phi_diff[np.where(phi_diff<-180)] += 360
        phi_diff[np.where(phi_diff>180)] -= 360
        mae_phi = np.abs(phi_diff)
        
        psi_diff = labels_[:,1] - predictions_[:,1]
        psi_diff[np.where(psi_diff<-180)] += 360
        psi_diff[np.where(psi_diff>180)] -= 360                          
        mae_psi = np.abs(psi_diff)
    
        return mae_phi, mae_psi
    
def clean_inputs(x, x_mask, dim_input):
    # set 0
    # x.shape: batch, seq_len, dim_input
    # x_mask.shape: batch, seq_len
    x_mask = tf.tile(x_mask[:,:,tf.newaxis], [1, 1, dim_input])
    x_clean = tf.where(tf.math.equal(x_mask, 0), x, x_mask-1)
    return x_clean

def get_output(name, predictions, x_mask, total_len):
    
    if name == "SS":
        
        ss_outputs = []
        
        ss_prediction = tf.nn.softmax(predictions[0])
        for i in predictions[1:]:
            ss_prediction += tf.nn.softmax(i)
        ss_prediction = tf.nn.softmax(ss_prediction)
        
        x_mask = x_mask.numpy()
        ss_prediction = ss_prediction.numpy()
        
        max_length = x_mask.shape[1]
        for i in range(x_mask.shape[0]):
            indiv_length = int(max_length-np.sum(x_mask[i]))
            ss_outputs.append(ss_prediction[i][:indiv_length])
        
        ss_outputs_concat = np.concatenate(ss_outputs, 0)
        assert ss_outputs_concat.shape[0] == total_len
        
        return ss_outputs, ss_outputs_concat
    
    elif name == "PhiPsi":
        
        phi_predictions = []
        psi_predictions = []
        phi_outputs = []
        psi_outputs = []
        for i in predictions:
            
            # i.shape: batch, seq_len, 4
            i = i.numpy()
            
            phi_prediction = np.zeros((i.shape[0], i.shape[1], 1))
            psi_prediction = np.zeros((i.shape[0], i.shape[1], 1))

            phi_prediction[:,:,0] = np.rad2deg(np.arctan2(i[:,:,0], i[:,:,1]))
            psi_prediction[:,:,0] = np.rad2deg(np.arctan2(i[:,:,2], i[:,:,3]))
            
            phi_predictions.append(phi_prediction)
            psi_predictions.append(psi_prediction)
        
        phi_predictions = np.concatenate(phi_predictions, -1)
        phi_predictions = np.median(phi_predictions, -1)

        psi_predictions = np.concatenate(psi_predictions, -1)
        psi_predictions = np.median(psi_predictions, -1)
        
        x_mask = x_mask.numpy()
        max_length = x_mask.shape[1]
        for i in range(x_mask.shape[0]):
            indiv_length = int(max_length-np.sum(x_mask[i]))
            phi_outputs.append(phi_predictions[i][:indiv_length])
            psi_outputs.append(psi_predictions[i][:indiv_length])
        
        phi_outputs_concat = np.concatenate(phi_outputs, 0)
        psi_outputs_concat = np.concatenate(psi_outputs, 0)
        assert phi_outputs_concat.shape[0] == psi_outputs_concat.shape[0] == total_len        
        
        return phi_outputs, psi_outputs, [phi_outputs_concat, psi_outputs_concat]