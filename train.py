# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""
import time
from my_model import Model
import tensorflow as tf
import numpy as np
from tensorflow import keras
from utils import InputReader, cal_accurarcy

if __name__ == '__main__':
    
    #parameters of training
    batch_size = 4
    epochs = 40
    early_stop = 4
    input_normalization = True
    learning_rate = 1e-3
    
    params = {}
    params["d_input"] = 76
    params["d_ss8_output"] = 8
    params["d_ss3_output"] = 3
    params["d_phipsi_output"] = 4
    params["d_csf_output"] = 3
    params["d_asa_output"] = 1
    params["d_rota_output"] = 8
    params["dropout_rate"] = 0.25
  
    #parameters of transfomer model
    params["transfomer_layers"] = 2
    params["transfomer_num_heads"] = 4
    
    #parameters of birnn model
    params["lstm_layers"] = 4
    params["lstm_units"] = 1024
    
    #parameters of cnn model
    params["cnn_layers"] = 5
    params["cnn_channels"] = 32
    
    params["save_path"] = r'./models'
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), len(logical_gpus))

    train_list_path = "/data/xugang/opus_contact/SPOT-1D/dataset/SPOT-1D-dataset/opus_dataset/clean/list_train"
    val_list_path = "/data/xugang/opus_contact/SPOT-1D/dataset/SPOT-1D-dataset/opus_dataset/clean/list_val"
    test_list_path = "/data/xugang/opus_contact/SPOT-1D/dataset/SPOT-1D-dataset/opus_dataset/clean/list_test2016"
    
    inputs_files_path = "/data/xugang/opus_contact/SPOT-1D/dataset/SPOT-1D-dataset/opus_dataset/clean/inputs"
    labels_files_path = "/data/xugang/opus_contact/SPOT-1D/dataset/SPOT-1D-dataset/opus_dataset/clean/labels" 
    
    model_c5 = Model(params=params, name="c5")
    
    train_reader = InputReader(data_list=train_list_path,
                               inputs_files_path=inputs_files_path,
                               labels_files_path=labels_files_path,
                               num_batch_size=batch_size,
                               input_norm=input_normalization, 
                               shuffle=True,
                               data_enhance=True)
    
    val_reader = InputReader(data_list=val_list_path,
                             inputs_files_path=inputs_files_path,
                             labels_files_path=labels_files_path,
                             num_batch_size=batch_size,
                             input_norm=input_normalization, 
                             shuffle=False,
                             data_enhance=False)
        
    test_reader = InputReader(data_list=test_list_path,
                              inputs_files_path=inputs_files_path,
                              labels_files_path=labels_files_path,
                              num_batch_size=batch_size,
                              input_norm=input_normalization, 
                              shuffle=False,
                              data_enhance=False)
    
    lr = tf.Variable(tf.constant(learning_rate), name='lr', trainable=False)
    optimizer = keras.optimizers.Adam(lr=lr)

    def train_step(x, x_mask, y, y_mask):
        
        ss8_predictions = ss3_predictions = phipsi_predictions = \
            csf_predictions = asa_predictions = rota_predictions = None

        with tf.GradientTape() as tape:
            ss8_predictions, ss3_predictions, phipsi_predictions, \
                csf_predictions, asa_predictions, rota_predictions, loss = \
                model_c5.inference(x, x_mask, y, y_mask, training=True)    
            
        trainable_variables = model_c5.transformer.trainable_variables + \
            model_c5.cnn.trainable_variables + model_c5.birnn.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, trainable_variables))
        
        return loss, ss8_predictions, ss3_predictions, phipsi_predictions, \
            csf_predictions, asa_predictions, rota_predictions

    def infer_step(x, x_mask):
        
        ss8_predictions = ss3_predictions = phipsi_predictions = \
            csf_predictions = asa_predictions = rota_predictions = None
            
        ss8_predictions, ss3_predictions, phipsi_predictions, \
            csf_predictions, asa_predictions, rota_predictions, _ = \
            model_c5.inference(x, x_mask, y, y_mask, training=False)
            
        return ss8_predictions, ss3_predictions, phipsi_predictions, \
            csf_predictions, asa_predictions, rota_predictions
    
    best_val_acc = 0
    for epoch in range(epochs):
        
        #======================Train======================
        accuracy_train_ss8 = []
        accuracy_train_ss3 = []
        accuracy_train_phi = []
        accuracy_train_psi = []
        for step, filenames_batch in enumerate(train_reader.dataset):
            start_time = time.time()
            # x (batch, max_len, 76)
            # x_mask (batch, max_len)
            # encoder_padding_mask (batch, 1, 1, max_len)
            # y (batch, max_len, 30)
            # y_mask (batch, max_len, 30)
            filenames, x, x_mask, y, y_mask, inputs_total_len, labels_total_len = \
                train_reader.read_file_from_disk(filenames_batch)
            
            assert inputs_total_len == labels_total_len

            loss, ss8_predictions, ss3_predictions, phipsi_predictions, \
                csf_predictions, asa_predictions, rota_predictions = \
                    train_step(x, x_mask, y, y_mask)
            
            accuracy_train_ss8.extend(
                cal_accurarcy("SS8", ss8_predictions, y, y_mask, total_len=inputs_total_len))
            
            accuracy_train_ss3.extend(
                cal_accurarcy("SS3", ss3_predictions, y, y_mask, total_len=inputs_total_len))
            
            mae_phi, mae_psi = cal_accurarcy("PhiPsi", phipsi_predictions, y, y_mask, total_len=inputs_total_len)
            accuracy_train_phi.extend(mae_phi)
            accuracy_train_psi.extend(mae_psi)
            
            run_time = time.time() - start_time
            
            if step % 10 == 0:
                print('Epoch: %d, step: %d, loss: %3.3f, acc8: %3.4f, acc3: %3.4f, phi: %3.2f, psi: %3.2f, time: %3.3f'
                      % (epoch, step, loss, np.mean(accuracy_train_ss8), np.mean(accuracy_train_ss3), 
                          np.mean(accuracy_train_phi), np.mean(accuracy_train_psi), run_time)) 

        #======================Val======================
        accuracy_val_ss8 = []
        accuracy_val_ss3 = []
        accuracy_val_phi = []
        accuracy_val_psi = []
        start_time = time.time()
        for step, filenames_batch in enumerate(val_reader.dataset):
            
            filenames, x, x_mask, y, y_mask, inputs_total_len, labels_total_len = \
                val_reader.read_file_from_disk(filenames_batch)
            
            assert inputs_total_len == labels_total_len

            ss8_predictions, ss3_predictions, phipsi_predictions, \
                csf_predictions, asa_predictions, rota_predictions = \
                    infer_step(x, x_mask)
                    
            accuracy_val_ss8.extend(
                cal_accurarcy("SS8", ss8_predictions, y, y_mask, total_len=inputs_total_len))

            accuracy_val_ss3.extend(
                cal_accurarcy("SS3", ss3_predictions, y, y_mask, total_len=inputs_total_len))

            mae_phi, mae_psi = cal_accurarcy("PhiPsi", phipsi_predictions, y, y_mask, total_len=inputs_total_len)
            accuracy_val_phi.extend(mae_phi)
            accuracy_val_psi.extend(mae_psi)
            
        run_time = time.time() - start_time
        print('Epoch: %d, lr: %s, acc8: %3.4f, acc3: %3.4f, phi: %3.2f, psi: %3.2f, time: %3.3f'
              % (epoch, str(lr.numpy()), np.mean(accuracy_val_ss8), np.mean(accuracy_val_ss3), 
                  np.mean(accuracy_val_phi), np.mean(accuracy_val_psi), run_time))   
        
        if np.mean(accuracy_val_ss8) > best_val_acc:
            best_val_acc = np.mean(accuracy_val_ss8)
            model_c5.save_model()
        else:
            lr.assign(lr/2)
            early_stop -= 1
        
        if early_stop == 0:
            break
    
    print ("best_val_acc:", best_val_acc)
    
    #======================Test======================

    model_c5_test = Model(params=params, name="c5")
    model_c5_test.load_model()
    
    def test_infer_step(x, x_mask):

        ss8_predictions = ss3_predictions = phipsi_predictions = \
            csf_predictions = asa_predictions = rota_predictions = None
            
        ss8_predictions, ss3_predictions, phipsi_predictions, \
            csf_predictions, asa_predictions, rota_predictions, _ = \
            model_c5_test.inference(x, x_mask, y, y_mask, training=False)
            
        return ss8_predictions, ss3_predictions, phipsi_predictions, \
            csf_predictions, asa_predictions, rota_predictions
            
    accuracy_test_ss8 = []
    accuracy_test_ss3 = []
    accuracy_test_phi = []
    accuracy_test_psi = []
    start_time = time.time()
    for step, filenames_batch in enumerate(test_reader.dataset):
        
        filenames, x, x_mask, y, y_mask, inputs_total_len, labels_total_len = \
            test_reader.read_file_from_disk(filenames_batch)
        
        assert inputs_total_len == labels_total_len

        ss8_predictions, ss3_predictions, phipsi_predictions, \
            csf_predictions, asa_predictions, rota_predictions = \
                test_infer_step(x, x_mask)

        accuracy_test_ss8.extend(
            cal_accurarcy("SS8", ss8_predictions, y, y_mask, total_len=inputs_total_len))

        accuracy_test_ss3.extend(
            cal_accurarcy("SS3", ss3_predictions, y, y_mask, total_len=inputs_total_len))

        mae_phi, mae_psi = cal_accurarcy("PhiPsi", phipsi_predictions, y, y_mask, total_len=inputs_total_len)
        accuracy_test_phi.extend(mae_phi)
        accuracy_test_psi.extend(mae_psi)
            
    run_time = time.time() - start_time
    print('Acc8: %3.4f, Acc3: %3.4f, Phi: %3.2f, Psi: %3.2f, time: %3.3f'
          % (np.mean(accuracy_test_ss8), np.mean(accuracy_test_ss3), 
              np.mean(accuracy_test_phi), np.mean(accuracy_test_psi), run_time))           
