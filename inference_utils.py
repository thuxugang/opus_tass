# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd

#=============================================================================    

def get_psp_dict():
    resname_to_psp_dict = {}
    resname_to_psp_dict['G'] = [1,4,7]
    resname_to_psp_dict['A'] = [1,3,7]
    resname_to_psp_dict['V'] = [1,7,12]
    resname_to_psp_dict['I'] = [1,3,7,12]
    resname_to_psp_dict['L'] = [1,5,7,12]
    resname_to_psp_dict['S'] = [1,2,5,7]
    resname_to_psp_dict['T'] = [1,7,15]
    resname_to_psp_dict['D'] = [1,5,7,11]
    resname_to_psp_dict['N'] = [1,5,7,14]
    resname_to_psp_dict['E'] = [1,6,7,11]
    resname_to_psp_dict['Q'] = [1,6,7,14]
    resname_to_psp_dict['K'] = [1,5,6,7,10]
    resname_to_psp_dict['R'] = [1,5,6,7,13]
    resname_to_psp_dict['C'] = [1,7,8]
    resname_to_psp_dict['M'] = [1,6,7,9]
    resname_to_psp_dict['F'] = [1,5,7,16]
    resname_to_psp_dict['Y'] = [1,2,5,7,16]
    resname_to_psp_dict['W'] = [1,5,7,18]
    resname_to_psp_dict['H'] = [1,5,7,17]
    resname_to_psp_dict['P'] = [7,19]
    return resname_to_psp_dict

def get_pc7_dict():
    resname_to_pc7_dict = {'A': [-0.350, -0.680, -0.677, -0.171, -0.170, 0.900, -0.476],
                'C': [-0.140, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476],
                'D': [-0.213, -0.417, -0.281, -0.767, -0.900, -0.155, -0.635],
                'E': [-0.230, -0.241, -0.058, -0.696, -0.868, 0.900, -0.582],
                'F': [ 0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318],
                'G': [-0.900, -0.900, -0.900, -0.342, -0.179, -0.900, -0.900],
                'H': [ 0.384, 0.110, 0.138, -0.271, 0.195, -0.031, -0.106],
                'I': [ 0.900, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688],
                'K': [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265],
                'L': [ 0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053],
                'M': [ 0.110, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001],
                'N': [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529],
                'P': [ 0.247, -0.900, -0.294, 0.055, -0.010, -0.900, 0.106],
                'Q': [-0.230, -0.110, -0.020, -0.464, -0.276, 0.528, -0.371],
                'R': [ 0.105, 0.373, 0.466, -0.900, 0.900, 0.528, -0.371],
                'S': [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212],
                'T': [ 0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212],
                'V': [ 0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.900],
                'W': [ 0.479, 0.900, 0.900, 0.900, -0.209, 0.279, 0.529],
                'Y': [ 0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476]}
    return resname_to_pc7_dict

resname_to_psp_dict = get_psp_dict()
resname_to_pc7_dict = get_pc7_dict()
    
def read_pssm(fname,seq):
    num_pssm_cols = 44
    pssm_col_names = [str(j) for j in range(num_pssm_cols)]
    with open(fname,'r') as f:
        tmp_pssm = pd.read_csv(f,delim_whitespace=True,names=pssm_col_names).dropna().values[:,2:22].astype(float)
    if tmp_pssm.shape[0] != len(seq):
        raise ValueError('PSSM file is in wrong format or incorrect!')
    return tmp_pssm

def read_hhm(fname,seq):
    num_hhm_cols = 22
    hhm_col_names = [str(j) for j in range(num_hhm_cols)]
    with open(fname,'r') as f:
        hhm = pd.read_csv(f,delim_whitespace=True,names=hhm_col_names)
    pos1 = (hhm['0']=='HMM').idxmax()+3
    num_cols = len(hhm.columns)
    hhm = hhm[pos1:-1].values[:,:num_hhm_cols].reshape([-1,44])
    hhm[hhm=='*']='9999'
    if hhm.shape[0] != len(seq):
        raise ValueError('HHM file is in wrong format or incorrect!')
    return hhm[:,2:-12].astype(float)

def read_fasta(fasta_path):
    files = []
    f = open(fasta_path, 'r')
    tmp = []
    for i in f.readlines():
        line = i.strip()
        if line[0] == '>':
            tmp.append(line[1:])
        else:
            tmp.append(line)
            files.append(tmp)
            tmp = []
    f.close()      
    return files

def get_pssm(file, preparation_config):
    
    filename = file[0].split('.')[0]
    fasta_content = ">" + filename + '\n' + file[1]
    
    fasta_path = os.path.join(preparation_config["tmp_files_path"], filename+'.fasta')
    output_path = os.path.join(preparation_config["tmp_files_path"], filename+'.txt')
    pssm_path = os.path.join(preparation_config["tmp_files_path"], filename+'.pssm')
    
    f = open(fasta_path, 'w')
    f.writelines(fasta_content)
    f.close()     

    cmd = preparation_config["psiblast_path"] + " -num_threads " + str(preparation_config["num_threads"]) + " -query " + \
            fasta_path + " -db " + preparation_config["uniref90_path"] + " -out " + \
            output_path  + " -num_iterations 3 -out_ascii_pssm " + pssm_path
            
    print (cmd)    
    
    output = os.popen(cmd).read() 
    
    if os.path.exists(output_path):
        os.remove(output_path)

def get_hhm(file, preparation_config):
    
    filename = file[0].split('.')[0]
    fasta_content = ">" + filename + '\n' + file[1]
    
    fasta_path = os.path.join(preparation_config["tmp_files_path"], filename+'.fasta')
    a3m_path = os.path.join(preparation_config["tmp_files_path"], filename+'.a3m')
    hhm_path = os.path.join(preparation_config["tmp_files_path"], filename+'.hhm')
    hhr_path = os.path.join(preparation_config["tmp_files_path"], filename+'.hhr')
   
    if not os.path.exists(fasta_path):
        f = open(fasta_path, 'w')
        f.writelines(fasta_content)
        f.close()     

    cmd = preparation_config["hhblits_path"] + " -i " + fasta_path + \
            " -ohhm " + hhm_path + " -oa3m " + a3m_path + " -d " + preparation_config["uniclust30_path"] + \
            " -v 0 -maxres 40000 -cpu " + str(preparation_config["num_threads"]) + " -Z 0"
            
    print (cmd)    
    
    output = os.popen(cmd).read() 
    
    if os.path.exists(a3m_path):
        os.remove(a3m_path)
    if os.path.exists(hhr_path):
        os.remove(hhr_path)

def make_input(file, preparation_config):
    """
    20pssm + 30hhm + 7pc + 19psp
    """    
    filename = file[0].split('.')[0]
    fasta = file[1]   
    
    seq_len = len(fasta)

    pssm_path = os.path.join(preparation_config["tmp_files_path"], filename+'.pssm')
    hhm_path = os.path.join(preparation_config["tmp_files_path"], filename+'.hhm')
    input_path = os.path.join(preparation_config["tmp_files_path"], filename+'.inputs')
    
    pssm = read_pssm(pssm_path, fasta)
    hhm = read_hhm(hhm_path, fasta)
    
    pc7 = np.zeros((seq_len, 7))
    for i in range(seq_len):
        pc7[i] = resname_to_pc7_dict[fasta[i]]
    
    psp = np.zeros((seq_len, 19))
    for i in range(seq_len):
        psp19 = resname_to_psp_dict[fasta[i]]
        for j in psp19:
            psp[i][j-1] = 1
    
    input_data = np.concatenate((pssm, hhm, pc7, psp),axis=1)
    assert input_data.shape == (seq_len,76)
    np.savetxt(input_path, input_data, fmt="%.4f")

#=============================================================================    

def read_inputs(filenames, inputs_files_path):
    """
    20pssm + 30hhm + 7pc + 19psp
    """
    inputs_nopadding = []
    max_len = 0
    inputs_total_len = 0
    for filename in filenames:
        inputs_ = np.loadtxt((os.path.join(inputs_files_path, filename + ".inputs")))
        
        inputs_total_len += inputs_.shape[0]
        if inputs_.shape[0] > max_len:
            max_len = inputs_.shape[0]
        inputs_nopadding.append(inputs_)
    
    inputs_padding = np.zeros(shape=(len(filenames), max_len, 76))
    inputs_mask_padding = np.ones(shape=(len(filenames), max_len))

    for i in range(len(filenames)):
        inputs_padding[i,:inputs_nopadding[i].shape[0]] = inputs_nopadding[i]
        inputs_mask_padding[i,:inputs_nopadding[i].shape[0]] = 0
        
    #(hhm - 5000) / 1000
    inputs_padding[:,:,20:50] = (inputs_padding[:,:,20:50] - 5000)/1000
        
    return inputs_padding, inputs_mask_padding, inputs_total_len

class InputReader(object):

    def __init__(self, data_list, num_batch_size, inputs_files_path):

        self.data_list = data_list
        self.inputs_files_path = inputs_files_path
        self.dataset = tf.data.Dataset.from_tensor_slices(self.data_list).batch(num_batch_size)          
        
        print ("Data Size:", len(self.data_list)) 
    
    def read_file_from_disk(self, filenames_batch):
        
        filenames_batch = [bytes.decode(i) for i in filenames_batch.numpy()]
        inputs_batch, inputs_masks_batch, inputs_total_len = \
            read_inputs(filenames_batch, self.inputs_files_path)
        
        inputs_batch = tf.convert_to_tensor(inputs_batch, dtype=tf.float32)
        inputs_masks_batch= tf.convert_to_tensor(inputs_masks_batch, dtype=tf.float32)
        
        return filenames_batch, inputs_batch, inputs_masks_batch, inputs_total_len
            
#=============================================================================    

def get_ensemble_ouput(name, predictions, x_mask, total_len):
    
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

ss8_str = "CSTHGIEB"
ss8_dict = {}
for k,v in enumerate(ss8_str):
    ss8_dict[k] = v

ss3_str = "CHE"
ss3_dict = {}
for k,v in enumerate(ss3_str):
    ss3_dict[k] = v
    
def output_results(filenames, ss8_outputs, ss3_outputs, phi_outputs, psi_outputs, preparation_config):
    
    for filename, ss8_output, ss3_output, phi_output, psi_output in \
        zip(filenames, ss8_outputs, ss3_outputs, phi_outputs, psi_outputs):
        
        output_path = os.path.join(preparation_config["output_path"], filename+".opus")
        f = open(output_path, 'w')
        f.write("#\tSS3\tSS8\tPhi\tPsi\tP(3-C)\tP(3-H)\tP(3-E)\tP(8-C)\tP(8-S)\t(8-T)\tP(8-H)\tP(8-G)\tP(8-I)\tP(8-E)\tP(8-B)\n")
        
        assert ss8_output.shape[0] == ss3_output.shape[0] == phi_output.shape[0] == psi_output.shape[0]

        for idx, (ss8, ss3, phi, psi) in \
            enumerate(zip(ss8_output, ss3_output, phi_output, psi_output)):
            
            ss8_cls = ss8_dict[np.argmax(ss8)]
            ss3_cls = ss3_dict[np.argmax(ss3)]
            
            ss3*=100
            ss8*=100
            
            f.write('%i\t%s\t%s'%(idx+1,ss3_cls,ss8_cls))
            f.write('\t%3.2f\t%3.2f'%(phi, psi))
            f.write('\t%3.2f\t%3.2f\t%3.2f'%tuple(ss3))
            f.write('\t%3.2f\t%3.2f\t%3.2f\t%3.2f\t%3.2f\t%3.2f\t%3.2f\t%3.2f\n'%tuple(ss8))
        
        f.close()
    
    
    
    