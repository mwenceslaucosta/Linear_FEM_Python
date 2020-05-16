# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:19:36 2020

@author: mathe
"""

import numpy as np 

class Inload: 
    """
    Class to assembly load vector 
    Implemented only nodal force for now
    """
#-----------------------------------------------------------------------------            
    def __init__(self,mesh):

        self.load_vector=np.zeros(mesh.n_GDL_tot)
        self.assembly_Neumman_point(mesh)


#-----------------------------------------------------------------------------            
    def assembly_Neumman_point(self,mesh):
        """
        Method to assembly Neummann nodal forces
        """
        cont=0
        for i in mesh.Neumann_pt[2]:
            self.load_vector[i]=mesh.Neumann_pt[1][cont]
            cont+=1
          
    
    
            
            
            
            
        
        