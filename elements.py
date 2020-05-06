# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:00:29 2020

@author: mathe
"""

from abc import ABC, abstractmethod

class Elements(ABC):
    """
    Class elements 
   
    """
    @abstractmethod
    def N_element(self):
         """
          Abstract method to compute element interpolation matrix 
         """
         pass
    @abstractmethod
    def B_element(self):
        """
        Abstract method to compute element interpolation matrix derivative
        """
        pass 
    @abstractmethod
    def K_element(self):
        """
        Abstract method to compute element stiffiness matrix
        """
        pass
    @abstractmethod
    def strain_element(self):
        """
        Abstract method to compute element strain 
        """
        pass
    @abstractmethod
    def get_extrapolation_matrix(self):
        """
        Abstract method to get extrapolation matrix 
        """ 
        pass