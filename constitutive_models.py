# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:20:45 2020

@author: mathe
"""
import numpy as np

class linear_elasticity_iso_3D:
    """
     Isotropic linear elastic constitutive model
    """
    def __init__(self,element,config_material):
        self.tangent_modulus=np.zeros((6,6))
        self.stress_1=np.zeros((6,element.n_gauss))
        self.strain_1=np.zeros((6,element.n_gauss))
        self.stress_0=np.zeros((6,element.n_gauss))
        self.strain_0=np.zeros((6,element.n_gauss))
        self.elastic_modulus=config_material[0]
        self.poisson=config_material[1]
        
#-----------------------------------------------------------------------------# 
        
    def get_tangent_modulus(self,displacemente,element):
        """
        Method to compute tangen modulus
        """ 
        A=self.poisson/(1-self.poisson)
        B=(1-2*self.poisson)/(2*((1-self.poisson)))
        C=self.elastic_modulus*(1-self.poisson)/((1+self.poisson)*(1-2*self.poisson))
        
        self.tangent_modulus[:,:]=0
        self.tangent_modulus[0,0]=1; self.tangent_modulus[1,1]=1
        self.tangent_modulus[1,1]=1; self.tangent_modulus[2,2]=1
        self.tangent_modulus[0,1]=A; self.tangent_modulus[0,2]=A
        self.tangent_modulus[1,0]=A; self.tangent_modulus[1,2]=A
        self.tangent_modulus[2,0]=A; self.tangent_modulus[2,1]=A
        self.tangent_modulus[3,3]=B; self.tangent_modulus[4,4]=B
        self.tangent_modulus[5,5]=B
        
        self.tangent_modulus=self.tangent_modulus*C
#-----------------------------------------------------------------------------# 

        