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
        n_GDL_node=mesh.n_GDL_node_element
        if hasattr(mesh,"BC_Neumann_point_X_"):
            n_direc_GDL=0
            self.assembly_Neumman_point(n_GDL_node,mesh.BC_Neumann_point_X_,n_direc_GDL)
        if hasattr(mesh,"BC_Neumann_point_Y_"):
            n_direc_GDL=1
            self.assembly_Neumman_point(n_GDL_node,mesh.BC_Neumann_point_Y_,n_direc_GDL)
        if hasattr(mesh,"BC_Neumann_point_Z_"):
            n_direc_GDL=2
            self.assembly_Neumman_point(n_GDL_node,mesh.BC_Neumann_point_Z_,n_direc_GDL)

#-----------------------------------------------------------------------------            
    def assembly_Neumman_point(self,n_GDL_node,List_BC_Neumann,n_direc_GDL):
        """
        Method to assembly Neummann nodal forces
        """
        for i in range(len(List_BC_Neumann[0])):
            n_nodes_BC=len(List_BC_Neumann[0][i])
            for j in range(n_nodes_BC):
                n_GDL_global=(List_BC_Neumann[0][i][j])*n_GDL_node+n_direc_GDL
                force_val=List_BC_Neumann[1][i]/n_nodes_BC
                self.load_vector[n_GDL_global]+=force_val
    
    
            
            
            
            
        
        