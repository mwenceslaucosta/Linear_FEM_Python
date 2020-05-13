# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:27:54 2020

@author: mathe
"""
import numpy as np
from scipy.sparse import coo_matrix
#-----------------------------------------------------------------------------    
class Assembly_KGlob:
    def __init__(self,mesh):
        n_positions=mesh.n_GDL_el*mesh.n_GDL_el*mesh.n_elements
        self.coo_i=np.zeros(n_positions)
        self.coo_j=np.zeros(n_positions)
        self.coo_data=np.zeros(n_positions)
        self.coo_data_BC=np.zeros(n_positions)
    
    def KGlobal(self,element,mesh,material):
        """ 
        Function to assembly global stifiness matrix. 
        """
        nGDL_tot=mesh.n_GDL_tot
        cont3=0
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
                            #K_glob[GDL_1,GDL_2]+=element[M].Ke[cont_1-1,cont_2-1]
                            self.coo_i[cont3]=GDL_1
                            self.coo_j[cont3]=GDL_2
                            self.coo_data[cont3]=element[M].Ke[cont_1-1,cont_2-1]
                            self.coo_data_BC[cont3]=element[M].Ke[cont_1-1,cont_2-1]
                            cont3+=1
                                                                   
        coo_Kglob=coo_matrix((self.coo_data,(self.coo_i, self.coo_j)),shape=(nGDL_tot,nGDL_tot))
        coo_Kglob_BC=coo_matrix((self.coo_data_BC,(self.coo_i, self.coo_j)),shape=(nGDL_tot,nGDL_tot))
        
        self.KGlob_csr=coo_Kglob.tocsr()
        self.KGlob_csr_BC=coo_Kglob_BC.tocsr()
                 
#-----------------------------------------------------------------------------
        

        
             
         
        
