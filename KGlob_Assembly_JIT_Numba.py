# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:15:09 2020

@author:  Matheus Wenceslau 
    Assembly logic performed by Thyller Brapp   
"""
from numba import jit
import numpy as np 
from hexaedron_8nodes import B_and_Ke_elem

#-----------------------------------------------------------------------------
@jit(nopython=True,cache=True)
def KGlobal(n_elem,n_nodes_element,DOF,cont,Dirichlet_DOF,Dirichlet_values,
            force_vector,connectivity,load_subtraction,coo_i,coo_j,coo_data,
            coo_data_BC,gauss_coord,gauss_weight,elem_coord,
            jacobian,det_Jacobian,deri_phi_param,deri_phi_real,B_elem,B_t,
            B_Gauss,K_e,nodes,tang_modu,B_all_elem):
    """
    Function to assembly global stifiness matrix.  
    
    """
    n_BC=Dirichlet_DOF.shape[0]
    cont[2]=0
    for M in range(n_elem):
        cont[0]=0
        #u_elem=get_u_elem(connectivity[M,:],n_nodes_element,DOF,u_glob,u_elem)
        K_e,B_e=B_and_Ke_elem(gauss_coord,
                            gauss_weight,elem_coord,connectivity[M,:],
                            jacobian,det_Jacobian,deri_phi_param,deri_phi_real,
                            B_elem,B_t,B_Gauss,K_e,nodes,tang_modu) 
        B_all_elem[M*48:(M*48+48),:]=B_e[:,:]
           
        for i in range(n_nodes_element):
            for j in range(DOF):
                cont[0]+=+1
                cont[1]=1
                for k in range(n_nodes_element):
                    for l in range(DOF):
                        DOF1=connectivity[M,i]*DOF+j
                        DOF2=connectivity[M,k]*DOF+l
                        coo_i[cont[2]]=DOF1
                        coo_j[cont[2]]=DOF2
                        coo_data[cont[2]]=K_e[cont[0]-1,cont[1]-1]
                        coo_data_BC[cont[2]]=K_e[cont[0]-1,cont[1]-1]
                        #Imposing Dirichlet Boundary Conditions
                        index1 = np.searchsorted(Dirichlet_DOF, DOF1)
                        index2 = np.searchsorted(Dirichlet_DOF, DOF2)
                        if (index1<n_BC):                            
                            if (Dirichlet_DOF[index1]==DOF1):
                                coo_data_BC[cont[2]]=0
                                if DOF1==DOF2:
                                    coo_data_BC[cont[2]]=1  
                        elif DOF1!=DOF2 and (index2<n_BC): 
                            if (Dirichlet_DOF[index2]==DOF2):
                                coo_data_BC[cont[2]]=0
                        cont[1]+=1
                        cont[2]+=1  
    return coo_i,coo_j,coo_data,coo_data_BC,B_all_elem

#-----------------------------------------------------------------------------
