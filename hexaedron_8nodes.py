# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:51:18 2020

@author: mathe
"""
import numpy as np 
from numba import jit

@jit(nopython=True,cache=True)
def B_and_Ke_elem(gauss_coord,gauss_weight,elem_coord,connectivity,
                   jacobian,det_Jacobian,deri_phi_param,deri_phi_real,B_elem,
                   B_t,B_Gauss,Ke,nodes,tang_modu):
    

    """
    Function to calculate B_all, Ke_all and det of the Jacobian of all elements

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
    
    n_gauss=8
    n_nodes_elem=8
    
    #Gauss point coordinantes and gauss_weights
    gauss_coord,gauss_weight=get_gauss_parametric_coordinante_and_weight(gauss_coord,gauss_weight)    
    elem_coord=get_coordinantes_nodes_elem(elem_coord,n_nodes_elem,
                            connectivity,nodes)
    #Derivative of the interpolation function in relation real coordinates                
    deri_phi_real,det_Jacobian=Interpol_fun_derivative(gauss_coord,elem_coord,
                              jacobian,n_gauss,deri_phi_param,det_Jacobian,deri_phi_real)
    
    #B Matrix
    B_elem=B_matrix(n_gauss,B_elem,deri_phi_real) 
    #Ke Matrix    
    Ke=Ke_matrix(Ke,gauss_weight,n_gauss,det_Jacobian,
                            B_elem,B_Gauss,B_t,tang_modu)
    return Ke,B_elem

#-----------------------------------------------------------------------------

@jit(nopython=True,cache=True)
def Interpol_fun_derivative(gauss_coord,elem_coord,jacobian,n_gauss,deri_phi_param,
                            det_Jacobian,deri_phi_real):      
    cd_no1=elem_coord[0,:] 
    cd_no2=elem_coord[1,:]  
    cd_no3=elem_coord[2,:]  
    cd_no4=elem_coord[3,:] 
    cd_no5=elem_coord[4,:]  
    cd_no6=elem_coord[5,:]  
    cd_no7=elem_coord[6,:]  
    cd_no8=elem_coord[7,:] 
    deri_phi_real[:,:]=0
    #Jacobian 
    cont=0
    for n_g in range(n_gauss):
        r=gauss_coord[n_g,0]
        s=gauss_coord[n_g,1]
        t=gauss_coord[n_g,2]

        
        dp_rf1_r=(1-s)*(1-t) 
        dp_rf3_r=(1+s)*(1-t) 
        dp_rf5_r=(1-s)*(1+t)
        dp_rf7_r=(1+s)*(1+t)
   
        dp_rf1_s=(1-r)*(1-t)
        dp_rf3_s=(1+r)*(1-t)
        dp_rf5_s=(1-r)*(1+t)
        dp_rf7_s=(1+r)*(1+t)
   
        dp_rf1_t=(1-r)*(1-s)
        dp_rf2_t=(1+r)*(1-s)
        dp_rf3_t=(1+r)*(1+s)
        dp_rf4_t=(1-r)*(1+s)
        
        
        #dx_dr
        jacobian[0,0]=(1/8)*((-cd_no1[0]+cd_no2[0])*dp_rf1_r  
                    +(cd_no3[0]-cd_no4[0])*dp_rf3_r
                    +(-cd_no5[0]+cd_no6[0])*dp_rf5_r  
                    +(cd_no7[0]-cd_no8[0])*dp_rf7_r)       
        
        #dx_ds
        jacobian[1,0]=(1/8)*((-cd_no1[0]+cd_no4[0])*dp_rf1_s 
                    + (-cd_no2[0]+cd_no3[0])*dp_rf3_s
                   +(-cd_no5[0]+cd_no8[0])*dp_rf5_s  
                   +(-cd_no6[0]+cd_no7[0])*dp_rf7_s)
        #dx_dt
        jacobian[2,0]=(1/8)*((-cd_no1[0]+cd_no5[0])*dp_rf1_t 
                            + (-cd_no2[0]+cd_no6[0])*dp_rf2_t
                           + (-cd_no3[0]+cd_no7[0])*dp_rf3_t 
                           + (-cd_no4[0]+cd_no8[0])*dp_rf4_t)
        #dy_dr
        jacobian[0,1]=(1/8)*((-cd_no1[1]+cd_no2[1])*dp_rf1_r  
                    + (cd_no3[1]-cd_no4[1])*dp_rf3_r
                   +(-cd_no5[1]+cd_no6[1])*dp_rf5_r 
                   + (cd_no7[1]-cd_no8[1])*dp_rf7_r)
        #dy_ds
        jacobian[1,1]=(1/8)*((-cd_no1[1]+cd_no4[1])*dp_rf1_s 
                             + (-cd_no2[1]+cd_no3[1])*dp_rf3_s
                   +(-cd_no5[1]+cd_no8[1])*dp_rf5_s 
                   + (-cd_no6[1]+cd_no7[1])*dp_rf7_s)
        #dy_dt       
        jacobian[2,1]=(1/8)*((-cd_no1[1]+cd_no5[1])*dp_rf1_t
                             + (-cd_no2[1]+cd_no6[1])*dp_rf2_t
                   + (-cd_no3[1]+cd_no7[1])*dp_rf3_t 
                   + (-cd_no4[1]+cd_no8[1])*dp_rf4_t)
        #dz_dr       
        jacobian[0,2]=(1/8)*((-cd_no1[2]+cd_no2[2])*dp_rf1_r 
                             + (cd_no3[2]-cd_no4[2])*dp_rf3_r
                  +(-cd_no5[2]+cd_no6[2])*dp_rf5_r
                  + (cd_no7[2]-cd_no8[2])*dp_rf7_r)
        #dz_ds
        jacobian[1,2]=(1/8)*((-cd_no1[2]+cd_no4[2])*dp_rf1_s 
                             + (-cd_no2[2]+cd_no3[2])*dp_rf3_s
                   +(-cd_no5[2]+cd_no8[2])*dp_rf5_s 
                   + (-cd_no6[2]+cd_no7[2])*dp_rf7_s)
        #dz_dt      
        jacobian[2,2]=(1/8)*((-cd_no1[2]+cd_no5[2])*dp_rf1_t
                             + (-cd_no2[2]+cd_no6[2])*dp_rf2_t
                   + (-cd_no3[2]+cd_no7[2])*dp_rf3_t 
                   + (-cd_no4[2]+cd_no8[2])*dp_rf4_t)
                        
        #Assembling the matrix of derivatives of interpolation functions 
        #in relation to the parametric coordinates for each Gauss point. 
        deri_phi_param[:,:]=0
        
        deri_phi_param[0,0]=-dp_rf1_r/8
        deri_phi_param[0,1]=dp_rf1_r/8
        deri_phi_param[0,2]=dp_rf3_r/8
        deri_phi_param[0,3]=-dp_rf3_r/8
        deri_phi_param[0,4]= -dp_rf5_r/8
        deri_phi_param[0,5]=dp_rf5_r/8
        deri_phi_param[0,6]=dp_rf7_r/8
        deri_phi_param[0,7]= -dp_rf7_r/8
        
        deri_phi_param[1,0]=-dp_rf1_s/8
        deri_phi_param[1,1]=-dp_rf3_s/8
        deri_phi_param[1,2]=dp_rf3_s/8
        deri_phi_param[1,3]=dp_rf1_s/8
        deri_phi_param[1,4]=-dp_rf5_s/8
        deri_phi_param[1,5]=-dp_rf7_s/8
        deri_phi_param[1,6]=dp_rf7_s/8
        deri_phi_param[1,7]=dp_rf5_s/8
        
        deri_phi_param[2,0]=-dp_rf1_t/8
        deri_phi_param[2,1]=-dp_rf2_t/8
        deri_phi_param[2,2]=-dp_rf3_t/8
        deri_phi_param[2,3]=-dp_rf4_t/8
        deri_phi_param[2,4]= dp_rf1_t/8
        deri_phi_param[2,5]=dp_rf2_t/8
        deri_phi_param[2,6]=dp_rf3_t/8
        deri_phi_param[2,7]= dp_rf4_t/8
        #Derivative of the interpolation function in relation real 
        #coordinates
        deri_phi_real[cont:cont+3,:]=np.linalg.solve(jacobian,deri_phi_param)
        
        #det_Jacobian
        det_Jacobian[n_g]=np.linalg.det(jacobian)
        cont+=3

    return deri_phi_real,det_Jacobian
#-----------------------------------------------------------------------------

@jit(nopython=True,cache=True)
def B_matrix(n_gauss,B_elem,deri_phi_real):            
    #B matrix
    #B_ele[0:6,:]= B matrix of the first guass point
    #B_ele[6:12,:]= B matrix of the second guass point...and so on
    cont_B1=0
    cont_B2=0

    for ii in range(n_gauss):      
        cont_B3=0
        for jj in range(n_gauss):
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
def Ke_matrix(Ke,gauss_weight,n_gauss,det_Jacobian,B_elem,B_Gauss,B_t,tang_modu):
    
    cont_g=0 
    # cont_inter=0
    # n_inter=int(inter_var.shape[0]/(2*n_gauss))
    # n_s=int(stress.shape[0]/2)
    
    # inter_var_0=inter_var[0:n_s]
    # stress_0=stress[0:n_s]
    # strain_0=strain[0:n_s]
    
    # inter_var_1=inter_var[n_s::]
    # stress_1=stress[n_s::]
    # strain_1=strain[n_s::]
    Ke[:,:]=0
    for n_g in range(n_gauss):
        B_Gauss[:,:]=B_elem[cont_g:cont_g+6,:]
        B_t[:,:]=np.transpose(B_Gauss)
        
        #Material model call 
        # stress_gauss_0=np.copy(stress_0[cont_g:cont_g+6])
        # stress_gauss_1=np.copy(stress_1[cont_g:cont_g+6])
        # strain_gauss_0=np.copy(strain_0[cont_g:cont_g+6])
        # strain_gauss_1=np.copy(strain_1[cont_g:cont_g+6])
        
        # inter_var_gauss_0=np.copy(inter_var_0[cont_inter:cont_inter+n_inter])
        # inter_var_gauss_1=np.copy(inter_var_1[cont_inter:cont_inter+n_inter])
                        
        #Gauss quadrature 
        w_i=gauss_weight[n_g,0]
        w_j=gauss_weight[n_g,1]
        w_k=gauss_weight[n_g,2]           
        A=det_Jacobian[n_g]*w_i*w_j*w_k
        B_we=B_Gauss*A
        dot_B_t_D=np.dot(B_t,tang_modu)
        Ke=Ke+np.dot(dot_B_t_D,B_we)                                                                 
        
        #Updating variables 
        # stress_0[cont_g:cont_g+6]=stress_1[cont_g:cont_g+6]
        # strain_0[cont_g:cont_g+6]=strain_1[cont_g:cont_g+6]
        # inter_var_0[cont_inter:cont_inter+n_inter]=inter_var_1[cont_inter:cont_inter+n_inter]
        
        # stress_1[cont_g:cont_g+6]=stress_gauss_1
        # strain_1[cont_g:cont_g+6]=strain_gauss_1
        # inter_var_1[cont_inter:cont_inter+n_inter]=inter_var_gauss_1
               
        cont_g=cont_g+6
        # cont_inter=cont_inter+n_inter
        
    return Ke
#-----------------------------------------------------------------------------

@jit(nopython=True,cache=True)
def get_gauss_parametric_coordinante_and_weight(gauss_coor,gauss_weight):
    
    """
    Method to get parametric coordinates of the integration points.
    """
    cont=0
    weights=np.array([1,1,1])
    #Unitary weights in this type of integration. Made just to remember. 
    r=np.array([-3**(-1/2), 3**(-1/2)])
    s=np.array([-3**(-1/2), 3**(-1/2)])
    t=np.array([-3**(-1/2), 3**(-1/2)])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                gauss_coor[cont,0]=r[i]
                gauss_coor[cont,1]=s[j]
                gauss_coor[cont,2]=t[k]
                gauss_weight[cont,0]=weights[i]
                gauss_weight[cont,1]=weights[j]
                gauss_weight[cont,2]=weights[k]
                cont=cont+1
    return gauss_coor,gauss_weight
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

