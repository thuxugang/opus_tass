# -*- coding: utf-8 -*-
"""
Created on Wed May 11 08:43:50 2016

@author: Xu Gang
"""

import numpy as np
import PDBreader
import residue
import calCSF

if __name__ == "__main__":
    
    filename = "./1a0tP.pdb"
    outputname = "./1a0tP.csf"
    
    print (filename)
    
    atomsData = PDBreader.readPDB(filename) 
    residuesData = residue.getResidueData(atomsData) 
    res_seq = [i.resname for i in residuesData]
    
    res_len = len(res_seq)
    
    csf_data = calCSF.getCSFFeature(residuesData)
    
    assert csf_data.shape[0] == res_len
    np.savetxt(outputname, csf_data, fmt="%.2f")
        


    