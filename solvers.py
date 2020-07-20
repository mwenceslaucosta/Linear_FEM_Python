# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:25:58 2020

@author: Matheus
"""
from init_Vars import Init_Vars
from KGlob_Assembly_JIT_Numba import KGlobal
from scipy.sparse import coo_matrix,linalg
from inload import Inload 
import pos_processing
import numpy as np 


"""
Implemented solvers: 
    Static linear - Infinitesimal deformation: static_linear 
To do:
    Static non linear  - Infinitesimal deformation
    Explicit linear dynamics  - Infinitesimal deformation
    Implicit linear dynamics  - Infinitesimal deformation
"""

def static_linear(mesh,material_model,mat_prop,out_file_name):
    
    Vars=Init_Vars(mesh,material_model)
    
    #External force
    external_force=Inload(mesh)
    load_vector=external_force.load_vector
    
    #Global Arrays
    Init_Vars
    
    #Elementary stifiness (Ke) and derivative matrix (B)
    element=mesh.fun_elem
    
    #2D Analysis
    if mesh.DOF_node_elem==2:
        Ke_all_elem,B_all_elem=get_Ke_all_and_B_all(mesh,material_model,element,
                                          mat_prop,Vars,mesh.thickness_vector)
    #3D Analysis
    elif  mesh.DOF_node_elem==3:
        Ke_all_elem,B_all_elem=get_Ke_all_and_B_all(mesh,material_model,element,
                                          mat_prop,Vars)
        
    #Global Sparse Matrix
    (Vars.coo_i,Vars.coo_j,Vars.coo_data,
     Vars.coo_data_BC)=KGlobal(
                      mesh.n_elem,mesh.n_nodes_elem,mesh.DOF_node_elem,
                      Vars.cont,mesh.Dirichlet_DOF_sorted,
                      mesh.Dirichlet_values,load_vector,mesh.connectivity,
                      Vars.load_subtraction,Vars.coo_i,Vars.coo_j,
                      Vars.coo_data,Vars.coo_data_BC,Vars.K_e,
                      Vars.Ke_all_elem,mesh.DOF_elem) 
           
    coo_Kglob=coo_matrix((Vars.coo_data,(Vars.coo_i,Vars.coo_j)),shape=(mesh.DOF_tot,mesh.DOF_tot))
    coo_Kglob_BC=coo_matrix((Vars.coo_data_BC,(Vars.coo_i,Vars.coo_j)),shape=(mesh.DOF_tot,mesh.DOF_tot))
    KGlob_csr=coo_Kglob.tocsr()
    KGlob_csr_BC=coo_Kglob_BC.tocsr()
    
    #Imposing BC in load vector 
    load_subtraction=imposing_force_BC(mesh,KGlob_csr,KGlob_csr_BC,
                    external_force.load_vector,Vars.load_subtraction,Vars.cont)
  
    #Displacement 
    displacement,nothing=linalg.cg(KGlob_csr_BC,load_subtraction)
    result={}
    result['displacement']=displacement
    
    #Pos-processing 
    if mesh.DOF_node_elem==2:
        extrapol_matrix=mesh.fun_elem.get_extrapolate_matrix(Vars.N,Vars.phi_vec,Vars.gauss_coor)
    elif  mesh.DOF_node_elem==3:
        extrapol_matrix=mesh.fun_elem.get_extrapolate_matrix(Vars.N,Vars.phi_vec,Vars.gauss_coor,mesh.mesh_type)
   
    tang_modu=material_model.tg_modulus(Vars.tang_modu,mat_prop)
  
    stress_gauss,strain_gauss,stress_nodes,strain_nodes=pos_processing.pos_static_linear(
                Vars.B_all_elem,tang_modu,Vars.stress_gauss,Vars.strain_gauss,
                 mesh.connectivity,mesh.n_nodes_elem,
                 mesh.n_Gauss_elem,mesh.DOF_node_elem,displacement,Vars.u_elem,
                mesh.n_elem,material_model,Vars.stress_nodes,Vars.strain_nodes,
                Vars.cont_average,Vars.extrapol_vec_stress,
                Vars.extrapol_vec_strain,mesh.DOF_stress_strain,extrapol_matrix)
    
    result['stress_gauss']=stress_gauss
    result['strain_gauss']=strain_gauss
    result['stress_nodes']=stress_nodes
    result['strain_nodes']=strain_nodes
 
    #Saving results 
    pos_processing.save_results(mesh,displacement,stress_nodes,strain_nodes,out_file_name)
            
    return result
        
#-----------------------------------------------------------------------------#        
def imposing_force_BC(mesh,KGlob_csc,KGlob_csc_BC,load_vector,load_subtraction,cont):
                    
    """
    Function to impose Dirichlet BC on the load vector. 
       
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

#-----------------------------------------------------------------------------#     
#@jit(nopython=True,cache=True,parallel=True,nogil=True)     
def get_Ke_all_and_B_all(mesh,material_model,element,mat_prop,Vars,thickness_vector=0):
    
    
    connectivity=mesh.connectivity
    nodes=mesh.nodes
    mesh_type=mesh.mesh_type
    Ke_all_elem=Vars.Ke_all_elem
    B_all_elem=Vars.B_all_elem
    n_elem=mesh.n_elem
    DOF_elem=mesh.DOF_elem
    DOF_node_elem=mesh.DOF_node_elem
    n_Gauss_elem=mesh.n_Gauss_elem
    DOF_stress_strain=mesh.DOF_stress_strain
    tang_modu=material_model.tg_modulus(Vars.tang_modu,mat_prop)
    
    n_compnts_B=DOF_stress_strain*n_Gauss_elem
    K_elem=np.zeros((DOF_elem,DOF_elem))
    gauss_coord=np.zeros((n_Gauss_elem,DOF_node_elem))
    gauss_weight=np.zeros((n_Gauss_elem,DOF_node_elem))
    elem_coord=np.zeros((n_Gauss_elem,DOF_node_elem))
    jacobian=np.zeros((DOF_node_elem,DOF_node_elem))
    det_Jacobian=np.zeros(n_Gauss_elem)
    deri_phi_param=np.zeros((DOF_node_elem,n_Gauss_elem))
    deri_phi_real=np.zeros((DOF_elem,n_Gauss_elem))
    B_elem=np.zeros((n_compnts_B,DOF_elem))
    B_t=np.zeros((DOF_elem,DOF_stress_strain))
    B_Gauss=np.zeros((DOF_stress_strain,DOF_elem))
    for M in range(n_elem):

        
        #2D Analysis
        if mesh.DOF_node_elem==2:
            thickness=thickness_vector[M]
            K_elem,B_elem=element.B_and_Ke_elem(gauss_coord,
                                gauss_weight,elem_coord,connectivity[M,:],
                                jacobian,det_Jacobian,deri_phi_param,deri_phi_real,
                                B_elem,B_t,B_Gauss,nodes,tang_modu,K_elem,thickness)
        #3D Analysis 
        elif mesh.DOF_node_elem==3:
            mesh_type=mesh.mesh_type
            K_elem,B_elem=element.B_and_Ke_elem(gauss_coord,gauss_weight,elem_coord,connectivity[M,:],
                   jacobian,det_Jacobian,deri_phi_param,deri_phi_real,B_elem,
                   B_t,B_Gauss,nodes,tang_modu,mesh_type,K_elem)
        
        B_all_elem[M*n_compnts_B:(M*n_compnts_B+n_compnts_B),:]=B_elem[:,:]
        Ke_all_elem[M*DOF_elem:(M*DOF_elem+DOF_elem),:]=K_elem[:,:]
    

    return Ke_all_elem,B_all_elem