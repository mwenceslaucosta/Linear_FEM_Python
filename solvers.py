# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:25:58 2020

@author: mathe
"""
from init_Vars import Init_Vars
#from init_Vars import Init_Vars_VE
#from init_Vars import Init_Vars_P
from KGlob_Assembly_JIT_Numba import KGlobal
from scipy.sparse import coo_matrix,linalg
from inload import Inload 
import numpy as np 


"""
Implemented solvers: 
    Static linear - Infinitesimal deformation: static_linear 
To do:
    Static non linear  - Infinitesimal deformation
    Explicit linear dynamics  - Infinitesimal deformation
    Implicit linear dynamics  - Infinitesimal deformation
"""

def static_linear(mesh,material_model,mat_prop):
    
    Vars=Init_Vars(mesh,material_model)
    
    #External force
    external_force=Inload(mesh)
    load_vector=external_force.load_vector
    #Global Arrays
    Init_Vars
    #Tangent modulus
    tang_modu=material_model.tg_modulus(Vars.tang_modu,mat_prop)
    (Vars.coo_i,Vars.coo_j,Vars.coo_data,
     Vars.coo_data_BC,Vars.B_all_elem)=KGlobal(
                     mesh.n_elem,mesh.n_nodes_elem,mesh.DOF_node_elem,
                               Vars.cont,mesh.Dirichlet_DOF_sorted,
                      mesh.Dirichlet_values,load_vector,mesh.connectivity,
                      Vars.load_subtraction,Vars.coo_i,Vars.coo_j,
                      Vars.coo_data,Vars.coo_data_BC,
                      Vars.gauss_coor,Vars.gauss_weight,
                      Vars.elem_coor,Vars.jacobian,
                      Vars.det_Jacobian,Vars.deri_phi_param,
                      Vars.deri_phi_real,Vars.B_elem,Vars.B_t,
                      Vars.B_Gauss,Vars.Ke,mesh.nodes,tang_modu,Vars.B_all_elem) 
    
    #Global Sparse Matrix
    coo_Kglob=coo_matrix((Vars.coo_data,(Vars.coo_i,Vars.coo_j)),shape=(mesh.DOF_tot,mesh.DOF_tot))
    coo_Kglob_BC=coo_matrix((Vars.coo_data_BC,(Vars.coo_i,Vars.coo_j)),shape=(mesh.DOF_tot,mesh.DOF_tot))
    KGlob_csc=coo_Kglob.tocsr()
    KGlob_csc_BC=coo_Kglob_BC.tocsr()
    
    load_subtraction=imposing_force_BC(mesh,KGlob_csc,KGlob_csc_BC,
                    external_force.load_vector,Vars.load_subtraction,Vars.cont)
  
    #Displacement 
    displacement=linalg.spsolve(KGlob_csc_BC,load_subtraction)
    
    #Pos-processing 
    stress,strain=material_model.get_stress_and_strain(Vars.B_all_elem,tang_modu,
                  Vars.stress,Vars.strain,mesh.connectivity,mesh.n_nodes_elem,
                 mesh.n_Gauss_elem,mesh.DOF_node_elem,displacement,Vars.u_elem,
                mesh.n_elem)
    return displacement,stress,strain
        
#-----------------------------------------------------------------------------#        
def imposing_force_BC(mesh,KGlob_csc,KGlob_csc_BC,load_vector,load_subtraction,cont):
                    
    """
    Method to impose Dirichlet BC on the load vector. 
       
    """
    if np.all(mesh.Dirichlet_values==0):
        load_subtraction=load_vector
    else: 
        cont[0]=0
        for i in mesh.Dirichlet_DOF:
            BC_value=mesh.Dirichlet_values[cont[0]]
            KGlob_csc[i,i]=0
            load_subtraction+=KGlob_csc[:,i].toarray().reshape(-1)*(-BC_value)
            load_subtraction[i]=0
            cont[0]+=1
        load_subtraction=load_vector+load_subtraction
        
        cont[0]=0
        for i in mesh.Dirichlet_DOF:
            BC_value=mesh.Dirichlet_values[cont[0]]
            load_subtraction[i]=BC_value*KGlob_csc_BC[i,i]
            cont[0]+=1
    return load_subtraction
        