# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:27:54 2020

@author: mathe
"""
import numpy as np
from scipy.sparse import coo_matrix
#-----------------------------------------------------------------------------    
class Assembly_KGlob_and_BC:
    def __init__(self,mesh):
        n_positions=mesh.n_GDL_el*mesh.n_GDL_el*mesh.n_elements
        self.coo_i=np.zeros(n_positions)
        self.coo_j=np.zeros(n_positions)
        self.coo_data=np.zeros(n_positions)
        self.coo_data_BC=np.zeros(n_positions)
        self.load_subtraction=np.zeros(mesh.n_GDL_tot)
        #self.K_glob=np.zeros((mesh.n_GDL_tot,mesh.n_GDL_tot))
    
    def KGlobal(self,element,mesh,material,external_force):
        """ 
        Function to assembly global stifiness matrix. 
        """
        nGDL_tot=mesh.n_GDL_tot
        self.cont3=0
        #Running only symmetrical part 
        for M in range(len(element)):
            Dimension=element[M].GDL_no_element
            cont_1=0
            for i in range(element[M].n_nodes_element):
                for j in range(Dimension):
                    cont_1+=+1
                    cont_2=cont_1
                    for k in range(i,element[M].n_nodes_element):
                        if j==0:
                            #Chamada normal correndo todas as colunas                             
                            for l in range(Dimension): 
                                self.allocate_value_Kglob_and_BC(mesh,element,M,i,j,k,
                                            l,cont_1,cont_2,Dimension)
                                cont_2+=1
                            
                        elif j==1: 
                            if cont_2==cont_1:
                                for l in range(1,Dimension):
                                    self.allocate_value_Kglob_and_BC(mesh,element,M,i,
                                          j,k,l,cont_1,cont_2,Dimension)
                                    cont_2+=1
                            else:
                                for l in range(Dimension):
                                    self.allocate_value_Kglob_and_BC(mesh,element,M,i,
                                           j,k,l,cont_1,cont_2,Dimension)
                                    cont_2+=1
                            
                        elif j==2: 
                            if cont_2==cont_1:
                                for l in range(2,Dimension): #Forncer manualmente os valores
                                    self.allocate_value_Kglob_and_BC(mesh,element,M,i,
                                           j,k,l,cont_1,cont_2,Dimension)
                                    cont_2+=1
                            else:
                                for l in range(Dimension):
                                    self.allocate_value_Kglob_and_BC(mesh,element,M,i,
                                           j,k,l,cont_1,cont_2,Dimension)
                                    cont_2+=1
                                                                                                                                                        
        coo_Kglob=coo_matrix((self.coo_data,(self.coo_i, self.coo_j)),shape=(nGDL_tot,nGDL_tot))
        coo_Kglob_BC=coo_matrix((self.coo_data_BC,(self.coo_i, self.coo_j)),shape=(nGDL_tot,nGDL_tot))
        
        self.KGlob_csc=coo_Kglob.tocsr()
        self.KGlob_csc_BC=coo_Kglob_BC.tocsr()
        self.imposing_force_B(mesh,external_force)
#-----------------------------------------------------------------------------
        
    def allocate_value_Kglob_and_BC(self,mesh,element,M,i,j,k,l,cont_1,cont_2,Dimension):
        GDL_1=mesh.connectivity[M,i]*Dimension+j 
        GDL_2=mesh.connectivity[M,k]*Dimension+l 
#        self.K_glob[GDL_1,GDL_2]+=element[M].Ke[cont_1-1,cont_2-1]
#        if GDL_1!=GDL_2:
#            self.K_glob[GDL_2,GDL_1]=self.K_glob[GDL_1,GDL_2]
            
        self.coo_i[self.cont3]=GDL_1
        self.coo_j[self.cont3]=GDL_2
        self.coo_data[self.cont3]=element[M].Ke[cont_1-1,cont_2-1]
        self.coo_data_BC[self.cont3]=element[M].Ke[cont_1-1,cont_2-1]
        
        if GDL_1!=GDL_2:
            self.coo_i[self.cont3+1]=GDL_2
            self.coo_j[self.cont3+1]=GDL_1
            self.coo_data[self.cont3+1]=element[M].Ke[cont_1-1,cont_2-1]
            self.coo_data_BC[self.cont3+1]=element[M].Ke[cont_1-1,cont_2-1] 
            
            
        #Dirichlet Boundary Conditions
        if (GDL_1 in mesh.Dirichlet_ind[2]) or (GDL_2 in mesh.Dirichlet_ind[2]):
            if GDL_1==GDL_2:
                self.coo_data_BC[self.cont3]=1
#                self.K_glob[GDL_1,GDL_2]=1
            else: 
                self.coo_data_BC[self.cont3]=0
                self.coo_data_BC[self.cont3+1]=0
#                self.K_glob[GDL_1,GDL_2]=0
#                self.K_glob[GDL_2,GDL_1]=0
                
        if GDL_1!=GDL_2:
            self.cont3+=2
        else:
            self.cont3+=1
        

#-----------------------------------------------------------------------------
                
    def imposing_force_B(self,mesh,force_vector):
        """
        Method to impose Dirichlet BC on the vector force. 

        Parameters
        ----------
        mesh: FEM Mesh

        Returns
        -------
        None.

        """
        cont=0
        for i in mesh.Dirichlet_ind[2]:
            BC_value=mesh.Dirichlet_ind[1][cont]
            self.load_subtraction+=self.KGlob_csc[:,i].toarray().reshape(-1)*BC_value
            self.load_subtraction[i]=0
            cont+=1
        self.load_subtraction-=force_vector
        
        cont=0
        for i in mesh.Dirichlet_ind[2]:
            BC_value=mesh.Dirichlet_ind[1][cont]
            self.load_subtraction[i]=BC_value*self.KGlob_csc_BC[i,i]
            cont+=1
            

        
             
         
        
