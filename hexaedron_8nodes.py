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
                   B_t,B_Gauss,nodes,tang_modu,mesh_type,K_e):
    
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
    gauss_coord=get_gauss_parametric_coordinante(gauss_coord,mesh_type)
    #Unitary weights in this type of integration. Made just to remember. 
    gauss_weight[:,:]=1
    elem_coord=get_coordinantes_nodes_elem(elem_coord,n_nodes_elem,
                            connectivity,nodes)
    #Derivative of the interpolation function in relation real coordinates                
    deri_phi_real,det_Jacobian=Interpol_fun_derivative(gauss_coord,elem_coord,
                              jacobian,n_gauss,deri_phi_param,det_Jacobian,deri_phi_real,mesh_type)
    
    #B Matrix
    B_elem=B_matrix(n_gauss,B_elem,deri_phi_real) 
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
    cd_no5=elem_coord[4,:]  
    cd_no6=elem_coord[5,:]  
    cd_no7=elem_coord[6,:]  
    cd_no8=elem_coord[7,:] 
    deri_phi_real[:,:]=0
    
    if mesh_type=='Abaqus':
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
    elif mesh_type=='Salome':
        #Jacobian 
        cont=0
        for n_g in range(n_gauss):
            r=gauss_coord[n_g,0]
            s=gauss_coord[n_g,1]
            t=gauss_coord[n_g,2]
    
            
            dp_rf2_r=(1-s)*(1-t) #-2,+3
            dp_rf6_r=(1+s)*(1-t) #-6,+7
            dp_rf1_r=(1-s)*(1+t)  #-1,+4
            dp_rf5_r=(1+s)*(1+t) #-5,+8
       
            dp_rf2_s=(1-r)*(1-t) #-2,+6
            dp_rf3_s=(1+r)*(1-t) #-3,+7
            dp_rf1_s=(1-r)*(1+t) #-1,+5
            dp_rf4_s=(1+r)*(1+t) #-4,+8
       
            dp_rf1_t=(1-r)*(1-s) #+1,-2
            dp_rf3_t=(1+r)*(1-s) #-3,+4
            dp_rf5_t=(1-r)*(1+s) #5,-6
            dp_rf7_t=(1+r)*(1+s) #-7,+8
      
            
            
            #dx_dr
            jacobian[0,0]=(1/8)*((-cd_no2[0]+cd_no3[0])*dp_rf2_r  
                        +(-cd_no6[0]+cd_no7[0])*dp_rf6_r
                        +(-cd_no1[0]+cd_no4[0])*dp_rf1_r  
                        +(-cd_no5[0]+cd_no8[0])*dp_rf5_r)       
            
            #dx_ds
            jacobian[1,0]=(1/8)*((-cd_no2[0]+cd_no6[0])*dp_rf2_s 
                        + (-cd_no3[0]+cd_no7[0])*dp_rf3_s
                       +(-cd_no1[0]+cd_no5[0])*dp_rf1_s  
                       +(-cd_no4[0]+cd_no8[0])*dp_rf4_s)
            #dx_dt
            jacobian[2,0]=(1/8)*((+cd_no1[0]-cd_no2[0])*dp_rf1_t 
                                + (-cd_no3[0]+cd_no4[0])*dp_rf3_t
                               + (+cd_no5[0]-cd_no6[0])*dp_rf5_t 
                               + (-cd_no7[0]+cd_no8[0])*dp_rf7_t)
            #dy_dr
            jacobian[0,1]=(1/8)*((-cd_no2[1]+cd_no3[1])*dp_rf2_r  
                        +(-cd_no6[1]+cd_no7[1])*dp_rf6_r
                        +(-cd_no1[1]+cd_no4[1])*dp_rf1_r  
                        +(-cd_no5[1]+cd_no8[1])*dp_rf5_r)
            #dy_ds
            jacobian[1,1]=(1/8)*((-cd_no2[1]+cd_no6[1])*dp_rf2_s 
                        + (-cd_no3[1]+cd_no7[1])*dp_rf3_s
                       +(-cd_no1[1]+cd_no5[1])*dp_rf1_s  
                       +(-cd_no4[1]+cd_no8[1])*dp_rf4_s)
            #dy_dt       
            jacobian[2,1]=(1/8)*((+cd_no1[1]-cd_no2[1])*dp_rf1_t 
                                + (-cd_no3[1]+cd_no4[1])*dp_rf3_t
                               + (+cd_no5[1]-cd_no6[1])*dp_rf5_t 
                               + (-cd_no7[1]+cd_no8[1])*dp_rf7_t)
            #dz_dr       
            jacobian[0,2]=(1/8)*((-cd_no2[2]+cd_no3[2])*dp_rf2_r  
                        +(-cd_no6[2]+cd_no7[2])*dp_rf6_r
                        +(-cd_no1[2]+cd_no4[2])*dp_rf1_r  
                        +(-cd_no5[2]+cd_no8[2])*dp_rf5_r)
            #dz_ds
            jacobian[1,2]=(1/8)*((-cd_no2[2]+cd_no6[2])*dp_rf2_s 
                        + (-cd_no3[2]+cd_no7[2])*dp_rf3_s
                       +(-cd_no1[2]+cd_no5[2])*dp_rf1_s  
                       +(-cd_no4[2]+cd_no8[2])*dp_rf4_s)
            #dz_dt      
            jacobian[2,2]=(1/8)*((+cd_no1[2]-cd_no2[2])*dp_rf1_t 
                                + (-cd_no3[2]+cd_no4[2])*dp_rf3_t
                               + (+cd_no5[2]-cd_no6[2])*dp_rf5_t 
                               + (-cd_no7[2]+cd_no8[2])*dp_rf7_t)
                            
            #Assembling the matrix of derivatives of interpolation functions 
            #in relation to the parametric coordinates for each Gauss point. 
            deri_phi_param[:,:]=0
            
            deri_phi_param[0,0]=-dp_rf1_r/8
            deri_phi_param[0,1]=-dp_rf2_r/8
            deri_phi_param[0,2]=dp_rf2_r/8
            deri_phi_param[0,3]=dp_rf1_r/8
            deri_phi_param[0,4]=-dp_rf5_r/8        
            deri_phi_param[0,5]=-dp_rf6_r/8
            deri_phi_param[0,6]=dp_rf6_r/8
            deri_phi_param[0,7]=dp_rf5_r/8
            
            deri_phi_param[1,0]=-dp_rf1_s/8
            deri_phi_param[1,1]=-dp_rf2_s/8
            deri_phi_param[1,2]=-dp_rf3_s/8
            deri_phi_param[1,3]=-dp_rf4_s/8
            deri_phi_param[1,4]=dp_rf1_s/8
            deri_phi_param[1,5]=dp_rf2_s/8
            deri_phi_param[1,6]=dp_rf3_s/8
            deri_phi_param[1,7]=dp_rf4_s/8
            
            deri_phi_param[2,0]=dp_rf1_t/8
            deri_phi_param[2,1]=-dp_rf1_t/8
            deri_phi_param[2,2]=-dp_rf3_t/8
            deri_phi_param[2,3]=dp_rf3_t/8
            deri_phi_param[2,4]=dp_rf5_t/8
            deri_phi_param[2,5]=-dp_rf5_t/8
            deri_phi_param[2,6]=-dp_rf7_t/8
            deri_phi_param[2,7]=dp_rf7_t/8
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
def Ke_matrix(gauss_weight,n_gauss,det_Jacobian,B_elem,B_Gauss,B_t,tang_modu,K_e):
    # K_e=np.zeros((24,24))
    cont_g=0 
    K_e[:,:]=0
    for n_g in range(n_gauss):
        B_Gauss[:,:]=B_elem[cont_g:cont_g+6,:]
        B_t[:,:]=np.transpose(B_Gauss)                       
        #Gauss quadrature 
        w_i=gauss_weight[n_g,0]
        w_j=gauss_weight[n_g,1]
        w_k=gauss_weight[n_g,2]           
        A=det_Jacobian[n_g]*w_i*w_j*w_k
        B_we=B_Gauss*A
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
    cte=3**(-1/2)

    if mesh_type=='Abaqus':
        gauss_coor[0,0]=-cte  ; gauss_coor[0,1]=-cte; gauss_coor[0,2]=-cte
        gauss_coor[1,0]=cte  ; gauss_coor[1,1]=-cte ; gauss_coor[1,2]=-cte
        gauss_coor[2,0]=cte  ; gauss_coor[2,1]=cte  ;  gauss_coor[2,2]=-cte
        gauss_coor[3,0]=-cte  ; gauss_coor[3,1]=cte  ;  gauss_coor[3,2]=-cte
        gauss_coor[4,0]=-cte  ; gauss_coor[4,1]=-cte  ;  gauss_coor[4,2]=cte
        gauss_coor[5,0]=cte  ; gauss_coor[5,1]=-cte  ;  gauss_coor[5,2]=cte
        gauss_coor[6,0]=cte  ; gauss_coor[6,1]=cte  ;  gauss_coor[6,2]=cte
        gauss_coor[7,0]=-cte  ; gauss_coor[7,1]=cte  ;  gauss_coor[7,2]=cte
        
    elif mesh_type=='Salome':
    
        gauss_coor[0,0]=-cte  ; gauss_coor[0,1]=-cte; gauss_coor[0,2]=cte   
        gauss_coor[1,0]=-cte  ; gauss_coor[1,1]=-cte ; gauss_coor[1,2]=-cte   
        gauss_coor[2,0]=cte  ; gauss_coor[2,1]=-cte  ;  gauss_coor[2,2]=-cte       
        gauss_coor[3,0]=cte  ; gauss_coor[3,1]=-cte  ;  gauss_coor[3,2]=cte     
        gauss_coor[4,0]=-cte  ; gauss_coor[4,1]=cte  ;  gauss_coor[4,2]=cte     
        gauss_coor[5,0]=-cte  ; gauss_coor[5,1]=cte  ;  gauss_coor[5,2]=-cte    
        gauss_coor[6,0]=cte  ; gauss_coor[6,1]=cte  ;  gauss_coor[6,2]=-cte    
        gauss_coor[7,0]=cte  ; gauss_coor[7,1]=cte  ;  gauss_coor[7,2]=cte
    

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
        phi[0]=(1/8)*(1-r)*(1-s)*(1-t)
        phi[1]=(1/8)*(1+r)*(1-s)*(1-t)
        phi[2]=(1/8)*(1+r)*(1+s)*(1-t)
        phi[3]=(1/8)*(1-r)*(1+s)*(1-t)
        phi[4]=(1/8)*(1-r)*(1-s)*(1+t)
        phi[5]=(1/8)*(1+r)*(1-s)*(1+t)
        phi[6]=(1/8)*(1+r)*(1+s)*(1+t) 
        phi[7]=(1/8)*(1-r)*(1+s)*(1+t)
    
    elif mesh_type=='Salome':
        phi[0]=(1/8)*(1-r)*(1-s)*(1+t)
        phi[1]=(1/8)*(1-r)*(1-s)*(1-t)
        phi[2]=(1/8)*(1+r)*(1-s)*(1-t)
        phi[3]=(1/8)*(1+r)*(1-s)*(1+t)
        phi[4]=(1/8)*(1-r)*(1+s)*(1+t)
        phi[5]=(1/8)*(1-r)*(1+s)*(1-t)    
        phi[6]=(1/8)*(1+r)*(1+s)*(1-t)
        phi[7]=(1/8)*(1+r)*(1+s)*(1+t) 
    
    return phi

#-----------------------------------------------------------------------------# 
@jit(nopython=True,cache=True)   
def get_extrapolate_matrix(N,phi,gauss_coor,mesh_type):
    """
    Extrapolate_matrix
    """
    gauss_coor=get_gauss_parametric_coordinante(gauss_coor,mesh_type)
    for i in range(8):
        phi=get_phi(phi,gauss_coor[i,0],gauss_coor[i,1],gauss_coor[i,2],mesh_type)
        for j in range(8):
            N[i,j]=phi[j]
     
 
    return np.linalg.inv(N) 
#-----------------------------------------------------------------------------# 

@jit(nopython=True,cache=True)   
def extrapolate_stress_strain(stress_gauss,strain_gauss,
                         stress_nodes,strain_nodes,cont_average,connectivity,
                        extrapol_vec_stress,extrapol_vec_strain,DOF_stress_strain,
                        n_Gauss_elem,N,phi_vec,gauss_coor,mesh_type):
    extrapol_matrix=get_extrapolate_matrix(N,phi_vec,gauss_coor,mesh_type)
    for M in range(connectivity.shape[0]):       
        for N in range(DOF_stress_strain):
            extrapol_vec_strain[::1]=0
            extrapol_vec_stress[::1]=0
            for j in range(n_Gauss_elem):
                extrapol_vec_stress[j]=stress_gauss[M,j*DOF_stress_strain+N]
                extrapol_vec_strain[j]=strain_gauss[M,j*DOF_stress_strain+N]
            stress_extra=extrapol_matrix @ extrapol_vec_stress
            strain_extra=extrapol_matrix @ extrapol_vec_strain
            
            for n_node in range(n_Gauss_elem):
                node=connectivity[M,n_node]
                stress_nodes[node,N]=stress_extra[n_node]+stress_nodes[node,N]
                strain_nodes[node,N]=strain_extra[n_node]+strain_nodes[node,N]
                cont_average[node]=cont_average[node]+1
    cont_average=cont_average/DOF_stress_strain
    for i in range(stress_nodes.shape[0]):
        stress_nodes[i,:]=stress_nodes[i,:]/cont_average[i]
        strain_nodes[i,:]=strain_nodes[i,:]/cont_average[i]
    return stress_nodes,strain_nodes