# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:27:54 2020

@author: mathe
"""
import numpy as np
#-----------------------------------------------------------------------------    
def KGlobal(K_glob,element,mesh,material):
    """ 
    Function to assembly global stifiness matrix. 
    """
    K_glob[:,:]=0
    for M in range(len(element)):
        Dimension=element[M].GDL_no_element
        cont_1=0
        for i in range(element[M].n_nodes_element):
            for j in range(Dimension):
                cont_1+=+1
                cont_2=0
                for k in range(element[M].n_nodes_element):
                    for l in range(Dimension):
                        cont_2+=+1
                        GDL_1=mesh.connectivity[M,i]*Dimension+j 
                        GDL_2=mesh.connectivity[M,k]*Dimension+l 
                        K_glob[GDL_1,GDL_2]+=element[M].Ke[cont_1-1,cont_2-1]
    return K_glob
                 
#-----------------------------------------------------------------------------    
             
         
        
