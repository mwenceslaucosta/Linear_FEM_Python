# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:27:54 2020

@author: mathe
"""
import numpy as np
import sys 

#-----------------------------------------------------------------------------    

class Init_Vars:                                           
    def __init__(self,mesh,mat_model):
        
        #Initializing arrays used in the global stiffiness assembly
        n_positions=mesh.DOF_elem*mesh.DOF_elem*mesh.n_elem   
        self.coo_i=np.zeros(n_positions,dtype=np.int32)
        self.coo_j=np.zeros(n_positions,dtype=np.int32)
        self.coo_data=np.zeros(n_positions)
        self.coo_data_BC=np.zeros(n_positions)
        self.cont=np.zeros(3,dtype=np.int32)
        self.load_subtraction=np.zeros(mesh.DOF_tot)
        
        #Initializing arrays used in the elementary stiffiness 
        if mesh.DOF_node_elem==3:
            n_Voight=6
        elif mesh.DOF_node_elem==2:
            n_Voight=3
        else:
            sys.exit('Fatal error: Number of Voigth component do not match')  
        
        self.elem_coor=np.zeros((mesh.n_nodes_elem,mesh.DOF_node_elem))
        self.phi=np.zeros((mesh.n_nodes_elem,mesh.n_Gauss_elem))
        self.jacobian=np.zeros((mesh.DOF_node_elem,mesh.DOF_node_elem)) 
        self.det_Jacobian=np.zeros((mesh.n_Gauss_elem))
        self.deri_phi_real=np.zeros((mesh.DOF_node_elem*mesh.n_Gauss_elem,mesh.n_nodes_elem))
        self.deri_phi_param=np.zeros((mesh.DOF_node_elem,mesh.n_Gauss_elem))
        self.gauss_coor=np.zeros((mesh.n_nodes_elem,mesh.DOF_node_elem))
        self.gauss_weight=np.zeros((mesh.n_nodes_elem,mesh.DOF_node_elem))
        self.B_elem=np.zeros((n_Voight*mesh.n_Gauss_elem,mesh.DOF_elem))
        self.B_all_elem=np.zeros((n_Voight*mesh.n_Gauss_elem*mesh.n_elem,mesh.DOF_elem))
        self.Ke_all_elem=np.zeros((mesh.DOF_elem*mesh.n_elem,mesh.DOF_elem))
        self.B_t=np.zeros((mesh.n_nodes_elem*mesh.DOF_node_elem,n_Voight))
        self.K_e=np.zeros((mesh.DOF_elem,mesh.DOF_elem))
        self.B_Gauss=np.zeros((n_Voight,mesh.DOF_elem)) 
        self.N=np.zeros((mesh.n_nodes_elem,mesh.n_nodes_elem))
        self.phi_vec=N=np.zeros(mesh.n_nodes_elem)

        material_name=mat_model.__name__
        if material_name.endswith('3D') and mesh.DOF_node_elem==2:
            sys.exit('Fatal error: Constitutive model does not match mesh type')
        elif material_name.endswith('2D') and mesh.DOF_node_elem==3:
            sys.exit('Fatal error: Constitutive model does not match mesh type')
        
        # To do 
        if material_name == 'plastic_Mises_3D':
            self.stress=np.zeros((6*mesh.n_Gauss_elem*2,mesh.n_elem))
            self.strain=np.zeros((6*mesh.n_Gauss_elem*2,mesh.n_elem))
            self.inter_var=np.zeros((13*mesh.n_Gauss_elem*2,mesh.n_elem))
            self.tang_modu=np.zeros((6,6))
                        
        elif (material_name == 'linear_elasticity_iso_3D' or 
              material_name == 'plane_stress_lin_elast_iso_2D'):
            self.stress_gauss=np.zeros((mesh.n_elem,n_Voight*mesh.n_Gauss_elem))
            self.strain_gauss=np.zeros((mesh.n_elem,n_Voight*mesh.n_Gauss_elem))
            self.stress_nodes=np.zeros((mesh.n_nodes_glob,n_Voight)) 
            self.strain_nodes=np.zeros((mesh.n_nodes_glob,n_Voight)) 
            self.cont_average=np.zeros(mesh.n_nodes_glob)
            self.extrapol_vec_strain=np.zeros(mesh.n_nodes_elem)
            self.extrapol_vec_stress=np.zeros(mesh.n_nodes_elem)
            self.tang_modu=np.zeros((n_Voight,n_Voight))

        self.u_glob=np.zeros((n_Voight*mesh.n_Gauss_elem*2,mesh.n_elem))
        self.u_elem=np.zeros(mesh.DOF_node_elem*mesh.n_nodes_elem)
            
        
        
#-----------------------------------------------------------------------------
            

        
             
         
        
