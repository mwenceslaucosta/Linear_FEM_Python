# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:55:44 2020

@author: Matheus Wenceslau
"""
from numba import jit

"""
 Plane stress Isotropic linear elastic constitutive model 
"""

@jit(nopython=True,cache=True)
def tg_modulus(tangent_modulus,mat_prop):
    
    elastic_modulus=mat_prop[0]
    poisson=mat_prop[1]
    
    A=elastic_modulus/(1-poisson**2)
    B=(1-poisson)/2
    
    tangent_modulus[:,:]=0
    tangent_modulus[0,0]=1
    tangent_modulus[1,1]=1 
    tangent_modulus[0,1]=poisson
    tangent_modulus[1,0]=poisson
    tangent_modulus[2,2]=B
    
    return tangent_modulus*A
    
#-----------------------------------------------------------------------------# 

@jit(nopython=True,cache=True)    
def get_stress_and_strain(B_all_elem,tangent_modulus,stress,strain,connectivity,
                          n_nodes_element,n_gauss,DOF,u_glob,u_elem,n_elem):
    """ 
    Function to compute stress and strain in gauss points 
    """
    
    for elem in range(n_elem):
        u_elem=get_u_elem(connectivity[elem,:],n_nodes_element,DOF,u_glob,u_elem)
        B_elem=B_all_elem[elem*n_gauss*3:(elem*n_gauss*3+n_gauss*3),::1]    
        for gauss in range(n_gauss):
            #B_ele[0:3,:]= B matrix of the first guass point
            B_Gauss=B_elem[gauss*3:(gauss*3+3),::1]
            strain_gauss=B_Gauss @ u_elem
            stress_gauss=tangent_modulus @ strain_gauss
            strain[elem,gauss*3:(gauss*3+3)]=strain_gauss[:]
            stress[elem,gauss*3:(gauss*3+3)]=stress_gauss[:]
    
    return stress,strain 

#-----------------------------------------------------------------------------# 
    
@jit(nopython=True,cache=True)
def get_u_elem(connectivity,n_nodes_element,DOF,u_glob,u_elem):
    
    """
    Function to get element displacement
    """
    cont=0
    for i in range(n_nodes_element):
        for j in range(DOF):
            GDL=connectivity[i]*DOF+j
            u_elem[cont]=u_glob[GDL]
            cont=cont+1
    return u_elem