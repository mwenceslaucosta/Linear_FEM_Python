# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 18:45:48 2020

@author: mathe
"""

import numpy as np 
from numba import jit

@jit(nopython=True,cache=True)
def B_and_Ke_elem(gauss_coord,gauss_weight,elem_coord,connectivity,
                   jacobian,det_Jacobian,deri_phi_param,deri_phi_real,B_elem,
                   B_t,B_Gauss,nodes,tang_modu,mesh_type,K_e):
    
    """
    Function to calculate B_elem and K_elem 

    """
    B_elem[:,:]=0
    gauss_weight[:,:]=0
    jacobian[:,:]=0
    B_Gauss[:,:]=0
    elem_coord[:,:]=0
    gauss_coord[:,:]=0
    deri_phi_real[:,:]=0
    det_Jacobian[:]=0
    deri_phi_param[:,:]=0
    B_t[:,:]=0
    
    n_gauss=1
    n_nodes_elem=4
    
    #Gauss point coordinantes and gauss_weights
    gauss_coord=get_gauss_parametric_coordinante(gauss_coord,mesh_type)
    #Unitary weights in this type of integration. Made just to remember. 
    gauss_weight[:,:]=1
    elem_coord=get_coordinantes_nodes_elem(elem_coord,n_nodes_elem,
                            connectivity,nodes)
    #Derivative of the interpolation function in relation real coordinates                
    deri_phi_real,det_Jacobian=Interpol_fun_derivative(gauss_coord,elem_coord,
                              jacobian,n_gauss,deri_phi_param,det_Jacobian,deri_phi_real,mesh_type)
    
    #B Matrix
    B_elem=B_matrix(n_gauss,B_elem,deri_phi_real,n_nodes_elem) 
    #Ke Matrix    
    K_e=Ke_matrix(gauss_weight,n_gauss,det_Jacobian,
                            B_elem,B_Gauss,B_t,tang_modu,K_e)
    return K_e,B_elem

#-----------------------------------------------------------------------------

@jit(nopython=True,cache=True)
def Interpol_fun_derivative(gauss_coord,elem_coord,jacobian,n_gauss,deri_phi_param,
                            det_Jacobian,deri_phi_real,mesh_type):      
    cd_no1=elem_coord[0,:] 
    cd_no2=elem_coord[1,:]  
    cd_no3=elem_coord[2,:]  
    cd_no4=elem_coord[3,:] 

    deri_phi_real[:,:]=0
    
    if mesh_type=='Abaqus':
       pass
   
    elif mesh_type=='Salome':
        #Jacobian 
        cont=0
        for n_g in range(n_gauss):
            r=gauss_coord[n_g,0]
            s=gauss_coord[n_g,1]
            t=gauss_coord[n_g,2]
               
            #dx_dr
            jacobian[0,0]=-cd_no2[0]+cd_no1[0]            
            #dx_ds
            jacobian[1,0]=-cd_no2[0]+cd_no3[0]
            #dx_dt
            jacobian[2,0]=-cd_no2[0]+cd_no4[0]
            #dy_dr
            jacobian[0,1]=-cd_no2[1]+cd_no1[1]
            #dy_ds
            jacobian[1,1]=-cd_no2[1]+cd_no3[1]
            #dy_dt       
            jacobian[2,1]=-cd_no2[1]+cd_no4[1]
            #dz_dr       
            jacobian[0,2]=-cd_no2[2]+cd_no1[2] 
            #dz_ds
            jacobian[1,2]=-cd_no2[2]+cd_no3[2]
            #dz_dt      
            jacobian[2,2]=-cd_no2[2]+cd_no4[2]
                            
            #Assembling the matrix of derivatives of interpolation functions 
            #in relation to the parametric coordinates for each Gauss point. 
            deri_phi_param[:,:]=0
            
            deri_phi_param[0,0]=1
            deri_phi_param[0,1]=-1
            deri_phi_param[0,2]=0
            deri_phi_param[0,3]=0

            
            deri_phi_param[1,0]=0
            deri_phi_param[1,1]=-1
            deri_phi_param[1,2]=1
            deri_phi_param[1,3]=0

            
            deri_phi_param[2,0]=0
            deri_phi_param[2,1]=-1
            deri_phi_param[2,2]=0
            deri_phi_param[2,3]=1

            #Derivative of the interpolation function in relation real 
            #coordinates
            deri_phi_real[cont:cont+3,:]=np.linalg.solve(jacobian,deri_phi_param)
            
            #det_Jacobian
            det_Jacobian[n_g]=np.linalg.det(jacobian)
            cont+=3
        

    return deri_phi_real,det_Jacobian
#-----------------------------------------------------------------------------

@jit(nopython=True,cache=True)
def B_matrix(n_gauss,B_elem,deri_phi_real,n_nodes_elem):            
    #B matrix
    #B_ele[0:6,:]= B matrix of the first guass point
    #B_ele[6:12,:]= B matrix of the second guass point...and so on
    cont_B1=0
    cont_B2=0

    for ii in range(n_gauss):      
        cont_B3=0
        for jj in range(n_nodes_elem):
            B_elem[cont_B1,cont_B3]=deri_phi_real[cont_B2,jj]
            B_elem[cont_B1+4,cont_B3]=deri_phi_real[cont_B2+2,jj]
            B_elem[cont_B1+5,cont_B3]=deri_phi_real[cont_B2+1,jj]
        
            B_elem[cont_B1+1,cont_B3+1]=deri_phi_real[cont_B2+1,jj]
            B_elem[cont_B1+3,cont_B3+1]=deri_phi_real[cont_B2+2,jj]
            B_elem[cont_B1+5,cont_B3+1]=deri_phi_real[cont_B2,jj]
        
            B_elem[cont_B1+2,cont_B3+2]=deri_phi_real[cont_B2+2,jj]
            B_elem[cont_B1+3,cont_B3+2]=deri_phi_real[cont_B2+1,jj]
            B_elem[cont_B1+4,cont_B3+2]=deri_phi_real[cont_B2,jj]
            cont_B3=cont_B3+3
        cont_B1=cont_B1+6
        cont_B2=cont_B2+3          
    return B_elem
#-----------------------------------------------------------------------------

@jit(nopython=True,cache=True)
def Ke_matrix(gauss_weight,n_gauss,det_Jacobian,B_elem,B_Gauss,B_t,tang_modu,K_e):
    """   
    Element Stiffness Matrix
    """    
    cont_g=0 
    K_e[:,:]=0
    for n_g in range(n_gauss):
        B_Gauss[:,:]=B_elem[cont_g:cont_g+6,:]
        B_t[:,:]=np.transpose(B_Gauss)                       
        #Gauss quadrature 
        w=gauss_weight[n_g,0]         
        A=det_Jacobian[n_g]*w
        B_we=B_Gauss*A*1/6
        dot_B_t_D=np.dot(B_t,tang_modu)
        K_e+=np.dot(dot_B_t_D,B_we)                                                                                
        cont_g=cont_g+6
        
    return K_e
#-----------------------------------------------------------------------------

@jit(nopython=True,cache=True)
def get_gauss_parametric_coordinante(gauss_coor,mesh_type):
    
    """
    Method to get parametric coordinates of the integration points.
    """
    cte=0.25

    if mesh_type=='Abaqus':
       pass
        
    elif mesh_type=='Salome':
    
        gauss_coor[0,0]=cte  ; gauss_coor[0,1]=cte; gauss_coor[0,2]=cte   
    
    return gauss_coor

#-----------------------------------------------------------------------------# 

@jit(nopython=True,cache=True)               
def get_coordinantes_nodes_elem(elem_coord,n_nodes_element,
                                connectivity_el,nodes):
    """
    Method to allocate element coordinates
    """
    elem_coord[:,:]=0
    for i in range(n_nodes_element):
        n_no=connectivity_el[i]
        elem_coord[i,:]=nodes[n_no,:]
    return elem_coord
#-----------------------------------------------------------------------------# 
@jit(nopython=True,cache=True)   
def get_phi(phi,r,s,t,mesh_type):
    """
    Interpolate function
    """
    if mesh_type=='Abaqus':
        pass

    
    elif mesh_type=='Salome':
        phi[0]=r
        phi[1]=1-r-s-t
        phi[2]=s
        phi[3]=t

    
    return phi

#-----------------------------------------------------------------------------# 
@jit(nopython=True,cache=True)   
def get_extrapolate_matrix(N,phi,gauss_coor,mesh_type):
    """
    Extrapolate_matrix
    """
    #Element with constant stress and strain 
    pass
      
#-----------------------------------------------------------------------------# 
    
@jit(nopython=True,cache=True)   
def extrapolate_stress_strain(stress_gauss,strain_gauss,
                         stress_nodes,strain_nodes,cont_average,connectivity,
                        extrapol_vec_stress,extrapol_vec_strain,DOF_stress_strain,
                        n_Gauss_elem,N,phi_vec,gauss_coor,mesh_type):

    for M in range(connectivity.shape[0]):       
        for N in range(DOF_stress_strain):
            extrapol_vec_strain[::1]=0
            extrapol_vec_stress[::1]=0
            for j in range(4):
                extrapol_vec_stress[j]=stress_gauss[M,N]
                extrapol_vec_strain[j]=strain_gauss[M,N]

            
            for n_node in range(4):
                node=connectivity[M,n_node]
                stress_nodes[node,N]=extrapol_vec_stress[n_node]+stress_nodes[node,N]
                strain_nodes[node,N]=extrapol_vec_strain[n_node]+strain_nodes[node,N]
                cont_average[node]=cont_average[node]+1
    cont_average=cont_average/DOF_stress_strain
    for i in range(stress_nodes.shape[0]):
        stress_nodes[i,:]=stress_nodes[i,:]/cont_average[i]
        strain_nodes[i,:]=strain_nodes[i,:]/cont_average[i]
    return stress_nodes,strain_nodes