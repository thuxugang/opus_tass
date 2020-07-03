# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

from my_model import Model
import tensorflow as tf

#============================Parameters====================================
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
    
#============================Models====================================

model_c31 = Model(params=params, name="c3")
model_c31.params["save_path"] = "./models/c3_1"
model_c31.load_model() 

model_c32 = Model(params=params, name="c3")
model_c32.params["save_path"] = "./models/c3_2"
model_c32.load_model() 

model_c41 = Model(params=params, name="c4")
model_c41.params["save_path"] = "./models/c4_1"
model_c41.load_model()   

model_c42 = Model(params=params, name="c4")
model_c42.params["save_path"] = "./models/c4_2"
model_c42.load_model()   

model_c43 = Model(params=params, name="c4")
model_c43.params["save_path"] = "./models/c4_3"
model_c43.load_model()  

model_c51 = Model(params=params, name="c5")
model_c51.params["save_path"] = "./models/c5_1"
model_c51.load_model()

model_c52 = Model(params=params, name="c5")
model_c52.params["save_path"] = "./models/c5_2"
model_c52.load_model()  

model_pp1 = Model(params=params, name="pp")
model_pp1.params["save_path"] = "./models/pp_1"
model_pp1.load_model()  

model_pp2 = Model(params=params, name="pp")
model_pp2.params["save_path"] = "./models/pp_2"
model_pp2.load_model() 

model_pp3 = Model(params=params, name="pp")
model_pp3.params["save_path"] = "./models/pp_3"
model_pp3.load_model() 

model_pp4 = Model(params=params, name="pp")
model_pp4.params["save_path"] = "./models/pp_4"
model_pp4.load_model() 

def test_infer_step(x, x_mask):
    
    ss8_predictions = []
    ss3_predictions = []
    phipsi_predictions = []
    
    ss8_prediction, ss3_prediction, phipsi_prediction, _, _ = \
        model_c31.inference(x, x_mask, y=None, y_mask=None, training=False)        
    ss8_predictions.append(ss8_prediction)
    ss3_predictions.append(ss3_prediction)
    #phipsi_predictions.append(phipsi_prediction)

    ss8_prediction, ss3_prediction, phipsi_prediction, _, _ = \
        model_c32.inference(x, x_mask, y=None, y_mask=None, training=False)        
    ss8_predictions.append(ss8_prediction)
    ss3_predictions.append(ss3_prediction)
    #phipsi_predictions.append(phipsi_prediction)

    ss8_prediction, ss3_prediction, phipsi_prediction, _, _, _ = \
        model_c41.inference(x, x_mask, y=None, y_mask=None, training=False)        
    ss8_predictions.append(ss8_prediction)
    ss3_predictions.append(ss3_prediction)
    #phipsi_predictions.append(phipsi_prediction)

    ss8_prediction, ss3_prediction, phipsi_prediction, _, _, _ = \
        model_c42.inference(x, x_mask, y=None, y_mask=None, training=False)        
    ss8_predictions.append(ss8_prediction)
    ss3_predictions.append(ss3_prediction)
    phipsi_predictions.append(phipsi_prediction)

    ss8_prediction, ss3_prediction, phipsi_prediction, _, _, _ = \
        model_c43.inference(x, x_mask, y=None, y_mask=None, training=False)        
    ss8_predictions.append(ss8_prediction)
    ss3_predictions.append(ss3_prediction)
    phipsi_predictions.append(phipsi_prediction)
    
    ss8_prediction, ss3_prediction, phipsi_prediction, _, _, _, _ = \
        model_c51.inference(x, x_mask, y=None, y_mask=None, training=False)        
    ss8_predictions.append(ss8_prediction)
    ss3_predictions.append(ss3_prediction)
    #phipsi_predictions.append(phipsi_prediction)

    ss8_prediction, ss3_prediction, phipsi_prediction, _, _, _, _ = \
        model_c52.inference(x, x_mask, y=None, y_mask=None, training=False)        
    ss8_predictions.append(ss8_prediction)
    ss3_predictions.append(ss3_prediction)
    phipsi_predictions.append(phipsi_prediction)
    
    #=====================================================================
    
    phipsi_prediction, _ = \
        model_pp1.inference(x, x_mask, y=None, y_mask=None, training=False)        
    phipsi_predictions.append(phipsi_prediction)

    phipsi_prediction, _ = \
        model_pp2.inference(x, x_mask, y=None, y_mask=None, training=False)        
    phipsi_predictions.append(phipsi_prediction)

    phipsi_prediction, _ = \
        model_pp3.inference(x, x_mask, y=None, y_mask=None, training=False)        
    phipsi_predictions.append(phipsi_prediction)

    phipsi_prediction, _ = \
        model_pp4.inference(x, x_mask, y=None, y_mask=None, training=False)        
    phipsi_predictions.append(phipsi_prediction)
    
    return ss8_predictions, ss3_predictions, phipsi_predictions