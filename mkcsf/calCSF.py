# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:03:01 2016

@author: XuGang
"""

import numpy as np

def getCSFFeature(residuesData):
    
    csf_data = []
    res_len = len(residuesData)
    for idx in range(res_len):
        ref_id = idx + 3
        if ref_id >= res_len:
            csf_data.append(np.array([0,0,0]))
        else:
            csf_data.append(transCoordinate(residuesData[ref_id].atoms["CA"].position, \
                residuesData[ref_id].atoms["C"].position, residuesData[ref_id].atoms["O"].position, \
                residuesData[idx].atoms["C"].position))
       
    return np.around(csf_data, decimals=2)
        
def transCoordinate(atom_ca_ref, atom_c_ref, atom_o_ref, atom_c):
    
    ref = atom_ca_ref
    c_ref_new = atom_c_ref - ref
    o_ref_new = atom_o_ref - ref
    c_new = atom_c - ref
  
    #c-ca
    x_axis = c_ref_new/np.linalg.norm(c_ref_new)
    
    c_o = o_ref_new - c_ref_new
    
    #o-c 与 x_axis垂直
    y_axis = c_o - (x_axis.dot(c_o)/x_axis.dot(x_axis) * x_axis)
    y_axis = y_axis/np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis,y_axis)
    
    rotation_matrix = np.array([x_axis[0],y_axis[0],z_axis[0],0,x_axis[1],y_axis[1],z_axis[1],0,x_axis[2],y_axis[2],z_axis[2],0,0,0,0,1]).reshape(4,4)

    new = np.array([c_new[0],c_new[1],c_new[2],1]).dot(rotation_matrix)

    return np.array([new[0],new[1],new[2]])








