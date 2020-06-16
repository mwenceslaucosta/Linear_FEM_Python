# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:51:51 2020

@author: mathe
"""
import numpy as np
from numba import jit

#-----------------------------------------------------------------------------# 
"""
 Isotropic linear elastic constitutive model 
"""

@jit(nopython=True,cache=True)
def tg_modulus(tangent_modulus,mat_prop):
    
    elastic_modulus=mat_prop[0]
    poisson=mat_prop[1]
    
    A=poisson/(1-poisson)
    B=(1-2*poisson)/(2*((1-poisson)))
    C=elastic_modulus*(1-poisson)/((1+poisson)*(1-2*poisson))
    
    tangent_modulus[:,:]=0
    tangent_modulus[0,0]=1; tangent_modulus[1,1]=1
    tangent_modulus[1,1]=1; tangent_modulus[2,2]=1
    tangent_modulus[0,1]=A; tangent_modulus[0,2]=A
    tangent_modulus[1,0]=A; tangent_modulus[1,2]=A
    tangent_modulus[2,0]=A; tangent_modulus[2,1]=A
    tangent_modulus[3,3]=B; tangent_modulus[4,4]=B
    tangent_modulus[5,5]=B
    
    tangent_modulus=tangent_modulus*C

    return tangent_modulus
#-----------------------------------------------------------------------------# 

@jit(nopython=True,cache=True)    
def get_stress_and_strain(B_all_elem,tangent_modulus,stress,strain,connectivity,
                          n_nodes_element,n_gauss,DOF,u_glob,u_elem,n_elem):
    """ 
    Function to compute stress and strain in gauss points 
    """
    
    for elem in range(n_elem):
        u_elem=get_u_elem(connectivity[elem,:],n_nodes_element,DOF,u_glob,u_elem)
        B_elem=B_all_elem[elem*48:(elem*48+48),::1]    
        for gauss in range(n_gauss):
            #B_ele[0:6,:]= B matrix of the first guass point
            B_Gauss=B_elem[gauss*6:(gauss*6+6),::1]
            strain_gauss=B_Gauss @ u_elem
            stress_gauss=tangent_modulus @ strain_gauss
            strain[gauss*6:(gauss*6+6),elem]=strain_gauss[:]
            stress[gauss*6:(gauss*6+6),elem]=stress_gauss[:]
    
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