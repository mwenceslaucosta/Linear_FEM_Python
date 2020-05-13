# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:52:14 2020

@author: mathe
"""
import numpy as np

class Dirichlet_imposition:
    """
    Class to dirichlet boundary condition imposition. 
    Only independent boundary condition implemented. 
    Constraint boundary condition not implemented.
    """

#-----------------------------------------------------------------------------#
    def __init__(self,mesh):
        self.load_subtract_BC=np.zeros(mesh.n_GDL_tot)
        

    def imposing_Dirichlet_BC(self,Kglobal,element,mesh,force_vector):
        """ 
        Method to impose Dirichlet's conditions. 
        """
        nGDL_tot=mesh.n_GDL_tot
        for M in range(len(element)):
            Dimension=element[M].GDL_no_element
            cont_1=0
            for i in range(element[M].n_nodes_element):
                for j in range(Dimension):
                    cont_1+=+1
                    cont_2=0
                    for k in range(element[M].n_nodes_element):
                        for l in range(Dimension):
                            cont_2+=+1
                            GDL_1=mesh.connectivity[M,i]*Dimension+j 
                            GDL_2=mesh.connectivity[M,k]*Dimension+l 
                            
                            #checking if it belongs to the BC group
                            if j == 0 and hasattr(mesh,"BC_Dirichlet_X"):
                                list_BC=mesh.BC_Dirichlet_X
                                self.Dirichlet_BC_KG(Kglobal,GDL_1,GDL_2,list_BC)
                                
                            elif j == 1 and hasattr(mesh,"BC_Dirichlet_Y"):
                                list_BC=mesh.BC_Dirichlet_Y
                                self.Dirichlet_BC_KG(Kglobal,GDL_1,GDL_2,list_BC)
                                
                            elif j == 2 and hasattr(mesh,"BC_Dirichlet_Z"):
                                list_BC=mesh.BC_Dirichlet_Z
                                self.Dirichlet_BC_KG(Kglobal,GDL_1,GDL_2,list_BC)
                                
        self.imposing_force_BC(mesh,force_vector)
                            
#-----------------------------------------------------------------------------
       
    def Dirichlet_BC_KG(self,Kglobal,GDL_1,GDL_2,list_BC):
                
        for i in list_BC[2]:
            cont1=0
            if GDL_1 in i:
                if GDL_1==GDL_2:
                    Kglobal.KGlob_csr_BC[GDL_1,GDL_2]=1
                    self.load_subtract_BC[GDL_1]=0
                else: 
                    BC_value=list_BC[1][cont1]
                    #Remember that K is symmetrical
                    self.load_subtract_BC[GDL_2]+=BC_value*Kglobal.KGlob_csr[GDL_1,GDL_2]   
                    Kglobal.KGlob_csr_BC[GDL_1,GDL_2]=0
                    Kglobal.KGlob_csr_BC[GDL_2,GDL_1]=0
            cont1+=1

#-----------------------------------------------------------------------------
            
    def imposing_force_BC(self,mesh,force_vector):
        """ 
        Method to impose Dirichlet's conditions in force vector. 
        """
          #checking if it belongs to the BC group
        if  hasattr(mesh,"BC_Dirichlet_X"):
            list_BC=mesh.BC_Dirichlet_X
            self.Dirichlet_BC_force_vector(list_BC,force_vector)
                                
        elif hasattr(mesh,"BC_Dirichlet_Y"): 
            list_BC=mesh.BC_Dirichlet_Y
            self.Dirichlet_BC_force_vector(list_BC,force_vector)
                                
        elif hasattr(mesh,"BC_Dirichlet_Z"):
            list_BC=mesh.BC_Dirichlet_Z
            self.Dirichlet_BC_force_vector(list_BC,force_vector)

#-----------------------------------------------------------------------------
                
                
    def Dirichlet_BC_force_vector(self,list_BC,force_vector):
        self.load_subtract_BC=force_vector-self.load_subtract_BC
        cont1=0
        for i in list_BC[2]:
            BC_value=list_BC[1][cont1]
            if BC_value != 0:
                for j in i:
                    self.load_subtract_BC[j]=BC_value
            cont1+=1
       
#-----------------------------------------------------------------------------
    