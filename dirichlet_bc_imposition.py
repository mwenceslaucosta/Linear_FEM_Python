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

    def __init__(self,mesh,external_force):
        if hasattr(mesh,"BC_Dirichlet_X"):
            self.n_nodes_BC_X=0
            for i in mesh.BC_Dirichlet_X[0]:
                self.n_nodes_BC_X+=i.shape[0]
                
            self.old_rows_KG_X=np.zeros((self.n_nodes_BC_X,mesh.n_GDL_tot))
            self.old_columms_KG_X=np.zeros((mesh.n_GDL_tot,self.n_nodes_BC_X))
            self.old_load_X=np.zeros(self.n_nodes_BC_X)

        if hasattr(mesh,"BC_Dirichlet_Y"):
            self.n_nodes_BC_Y=0
            for i in mesh.BC_Dirichlet_Y[0]:
                self.n_nodes_BC_Y+=i.shape[0]
                
            self.old_rows_KG_Y=np.zeros((self.n_nodes_BC_Y,mesh.n_GDL_tot))
            self.old_columms_KG_Y=np.zeros((mesh.n_GDL_tot,self.n_nodes_BC_Y))
            self.old_load_Y=np.zeros(self.n_nodes_BC_Y)

        if hasattr(mesh,"BC_Dirichlet_Z"):
            self.n_nodes_BC_Z=0
            for i in mesh.BC_Dirichlet_Z[0]:
                self.n_nodes_BC_Z+=i.shape[0]
                
            self.old_rows_KG_Z=np.zeros((self.n_nodes_BC_Z,mesh.n_GDL_tot))
            self.old_columms_KG_Z=np.zeros((mesh.n_GDL_tot,self.n_nodes_BC_Z))    
            self.old_load_Z=np.zeros(self.n_nodes_BC_Z)
        self.load_subtraction=np.zeros(mesh.n_GDL_tot)
            
#-----------------------------------------------------------------------------# 

    def dirichlet_imposition(self,mesh,K_glob,external_force):
       
        if hasattr(mesh,"BC_Dirichlet_X"):
            self.old_rows_KG_X[:,:]=0
            self.old_columms_KG_X[:,:]=0
            self.load_subtraction[:]=0
            self.old_load_X[:]=0
            list_BC_Dirichlet_X=mesh.BC_Dirichlet_X
            
            self.old_rows_KG_X,self.old_columms_KG_X,self.old_load_X,\
             K_glob,external_force=self.get_vectors_with_conditions_imposed(\
                 self.old_rows_KG_X,self.old_columms_KG_X,self.load_subtraction,\
                      list_BC_Dirichlet_X,K_glob,external_force,self.old_load_X)      
                 
        if hasattr(mesh,"BC_Dirichlet_Y"):
            self.old_rows_KG_Y[:,:]=0
            self.old_columms_KG_Y[:,:]=0
            self.load_subtraction[:]=0
            self.old_load_Y[:]=0
            list_BC_Dirichlet_Y=mesh.BC_Dirichlet_Y
            
            self.old_rows_KG_Y,self.old_columms_KG_Y,self.old_load_Y,\
             K_glob,external_force=self.get_vectors_with_conditions_imposed(\
                 self.old_rows_KG_Y,self.old_columms_KG_Y,self.load_subtraction,\
                      list_BC_Dirichlet_Y,K_glob,external_force,self.old_load_Y) 

        if hasattr(mesh,"BC_Dirichlet_Z"):
            self.old_rows_KG_Z[:,:]=0
            self.old_columms_KG_Z[:,:]=0
            self.load_subtraction[:]=0
            self.old_load_Z[:]=0
            list_BC_Dirichlet_Z=mesh.BC_Dirichlet_Z
            
            self.old_rows_KG_Z,self.old_columms_KG_Z,self.old_load_Z,\
             K_glob,external_force=self.get_vectors_with_conditions_imposed(\
                 self.old_rows_KG_Z,self.old_columms_KG_Z,self.load_subtraction,\
                      list_BC_Dirichlet_Z,K_glob,external_force,self.old_load_Z) 
                    
        return K_glob,external_force
#-----------------------------------------------------------------------------# 
        
    def get_vectors_with_conditions_imposed(self,old_rows,old_columms,load_subtraction,
                            list_BC_Dirichlet,K_glob,external_force,old_load):
        cont1=0
        cont2=0
        for i in list_BC_Dirichlet[2]:
            BC_value=list_BC_Dirichlet[1][cont1]
            for j in i:
                #Armazening K_global values and external force   
                old_rows[cont2,:]=K_glob[j,:]
                old_columms[:,cont2]=K_glob[:,j]
                old_load[cont2]=external_force[j]
                load_subtraction[:]=K_glob[:,j]
                
                load_subtraction=load_subtraction*BC_value
                load_subtraction[j]=0
                external_force[j]=0
                K_glob[j,:]=0
                K_glob[:,j]=0
                external_force-=load_subtraction
                cont2+=1
            cont1+=1
                
        cont1=0
        cont2=0
        for i in list_BC_Dirichlet[2]:
            BC_value=list_BC_Dirichlet[1][cont1]
            for j in i:
                external_force[j]=BC_value
                K_glob[j,j]=1
                
        return old_rows,old_columms,old_load,K_glob,external_force
                                                            
        
#-----------------------------------------------------------------------------# 
    