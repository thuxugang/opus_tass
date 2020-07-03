# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import time
from inference_utils import read_fasta, get_pssm, get_hhm, make_input, \
                            InputReader, get_ensemble_ouput, output_results
from inference_models import test_infer_step

if __name__ == '__main__':
    
    #============================Parameters====================================
    fasta_path = r"./casp.fasta"
    
    preparation_config = {}
    preparation_config["tmp_files_path"] = r"./tmp_files"
    preparation_config["output_path"] = r"./predictions"
    
    preparation_config["num_threads"] = 40
    preparation_config["psiblast_path"] = r'/data/xugang/opus_contact/blast/ncbi-blast-2.10.0+/bin/psiblast'
    preparation_config["uniref90_path"] = r'/data/xugang/opus_contact/uniref90/uniref90.fasta'
    preparation_config["hhblits_path"] = r'/data/xugang/opus_contact/hhblits/hh-suite/build/bin/hhblits'
    preparation_config["uniclust30_path"] = r'/data/xugang/opus_contact/uniclust30/uniclust30_2018_08/uniclust30_2018_08'
    
    batch_size = 8
    #============================Parameters====================================
    
    
    #============================Preparation===================================
    start_time = time.time()
    filenames = []
    files = read_fasta(fasta_path) 
    for file in files:
        
        filename = file[0].split('.')[0]
        fasta = file[1]
        
        pssm_filename = filename + '.pssm'
        if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], pssm_filename)):
            get_pssm(file, preparation_config)
        
        hhm_filename = filename + '.hhm'
        if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], hhm_filename)):
            get_hhm(file, preparation_config)       
        
        make_input(file, preparation_config)
        filenames.append(filename)
    run_time = time.time() - start_time
    print('Preparation done..., time: %3.3f' % (run_time))  
    #============================Preparation===================================
    
    #==================================Model===================================
    start_time = time.time()
    test_reader = InputReader(data_list=filenames, 
                              num_batch_size=batch_size,
                              inputs_files_path=preparation_config["tmp_files_path"])
    
    total_lens = 0
    for step, filenames_batch in enumerate(test_reader.dataset):
        # x (batch, max_len, 76)
        # x_mask (batch, max_len)
        filenames, x, x_mask, inputs_total_len = \
            test_reader.read_file_from_disk(filenames_batch)
        
        total_lens += inputs_total_len
        
        ss8_predictions, ss3_predictions, phipsi_predictions = \
            test_infer_step(x, x_mask)
            
        ss8_outputs, _ = \
            get_ensemble_ouput("SS", ss8_predictions, x_mask, inputs_total_len)
            
        ss3_outputs, _ = \
            get_ensemble_ouput("SS", ss3_predictions, x_mask, inputs_total_len)
            
        phi_outputs, psi_outputs, _ = \
            get_ensemble_ouput("PhiPsi", phipsi_predictions, x_mask, inputs_total_len)    
        
        assert len(filenames) == len(ss8_outputs) == len(ss3_outputs) == \
            len(phi_outputs) == len(psi_outputs)
            
        output_results(filenames, ss8_outputs, ss3_outputs, phi_outputs, psi_outputs, preparation_config)
        
    run_time = time.time() - start_time
    print('Prediction done..., time: %3.3f' % (run_time)) 
    #==================================Model===================================
    
    
    