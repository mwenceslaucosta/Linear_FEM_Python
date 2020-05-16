# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:55:50 2020

@author: Matheus_Wenceslau
"""
import numpy as np
import meshio 
import sys 

class MeshFEM:
    """
    Class to import mesh in .inp format (Abaqus) from GMSH
    Element implemented: C3D8-Hexahedron 
    Only one type of element per analysis. 
    Parameters: 
    config_mesh: Dictionary to read and config mesh
    1- config_mesh['mesh_file_name']: Name of the mesh file in in .inp format 
    2-config_mesh['analysis_dimension']: 
        analysis_dimension=3D (Only implemented for now)
              or 
       analysis_dimension=2D_plane_stress
       
    3- vector containing the values of BC groups as detailed below
    
    Attention to the six possible names of the groups in the dic config_mesh:
        BC_Neumann_point_X_
        BC_Neumann_point_Y_     for nodal force (only implemented for now)
        BC_Neumann_point_Z_
        
        BC_Dirichlet_X_
        BC_Dirichlet_Y_
        BC_Dirichlet_Z_
        
        Example:config_mesh['BC_Neumann_point_X_']
        BC_Neumann_X_ --> 
        vector cointaning the nodal Neumman BC values in X direction of each group.
            example for 2 groups: config_mesh['BC_Neumann_point_X_']=[100 200]
   
    ----------------------------!!!ATTENTION!!!------------------------------
    The names of the Boundary Condition groups should follow the standard:
    Neuman BC in x direction: 
        Nodal force Group 0: BC_Neumann_point_X_0
        Nodal force Group 1: BC_Neumann_point_X_1 ....
        and so on for the Y and Z direction groups. 
        
    Dirichlet BC in x direction: 
        Group 0: BC_Dirichlet_X_0
        Group 1: BC_Dirichlet_X_1 ....
        and so on for the Y and Z direction groups.
    """
#-----------------------------------------------------------------------------    
    def __init__(self,config_mesh):
        """
        Constructor meshFEM class
        """
        self.mesh_file=config_mesh['mesh_file_name']
        read_mesh=meshio.read(self.mesh_file)
        self.analysis_dimension=config_mesh['analysis_dimension']        
        self.nodes=read_mesh.points
        self.n_nodes_glob=self.nodes.shape[0]
 #       self.lines=read_mesh.cells_dict['line']
        self.get_connectivity(read_mesh)
        self.get_BC(read_mesh,config_mesh)
      
#-----------------------------------------------------------------------------    
                           
    def get_BC(self,read_mesh,config_mesh):
        """
        Method to get the list of nodes and values of the Boundary Conditions(BC)
        
        direction_BC - Direction of the BC
        0 - for x 
        1 - for y 
        2 - for z 
        
        self.Neumann_pt[0]: Vector containing all nodes with nodal Neumman BC 
        self.Neumann_pt[1]: Vector containing BC value of each DOF
        self.Neumann_pt[2]: Vector containing nodal Neumman BC DOF.
        
        self.Dirichlet_ind[0]: Vector containing all nodes with Dirichlet BC 
        self.Dirichlet_ind[1]: Vector containing BC value of each DOF
        self.Dirichlet_ind[2]: Vector containing Dirichlet BC DOF.
        """
        cont=0
        #Neumann nodal
        BC_type='Neumman_point'
        n=self.get_number_BC_directions(config_mesh,BC_type)
        self.Neumann_pt=[[None]*n,[None]*n,[None]*n]
        cont_n=0
        if 'BC_Neumann_point_X_' in config_mesh:
            name_group='BC_Neumann_point_X_'
            direction_BC=0 
            self.Neumann_pt[0][cont_n],self.Neumann_pt[1][cont_n],\
            self.Neumann_pt[2][cont_n]=self.list_BC(read_mesh,name_group,config_mesh,direction_BC,BC_type)
            cont_n+=1
            cont+=1
        if 'BC_Neumann_point_Y_' in config_mesh:
            name_group='BC_Neumann_point_Y_'
            direction_BC=1
            self.Neumann_pt[0][cont_n],self.Neumann_pt[1][cont_n],\
            self.Neumann_pt[2][cont_n]=self.list_BC(read_mesh,name_group,config_mesh,direction_BC,BC_type)
            cont_n+=1
            cont+=1
        if 'BC_Neumann_point_Z_' in config_mesh:
            name_group='BC_Neumann_point_Z_'
            direction_BC=2
            self.Neumann_pt[0][cont_n],self.Neumann_pt[1][cont_n],\
            self.Neumann_pt[2][cont_n]=self.list_BC(read_mesh,name_group,config_mesh,direction_BC,BC_type)
            cont_n+=1
            cont+=1
            
        self.Neumann_pt[0]=np.concatenate(self.Neumann_pt[0]) 
        self.Neumann_pt[1]=np.concatenate(self.Neumann_pt[1])
        self.Neumann_pt[2]=np.concatenate(self.Neumann_pt[2])
        
        #Dirichlet
        BC_type='Dirichlet_ind'
        n=self.get_number_BC_directions(config_mesh,BC_type)
        self.Dirichlet_ind=[[None]*n,[None]*n,[None]*n]
        cont_n=0
        if 'BC_Dirichlet_X_' in config_mesh:
            name_group='BC_Dirichlet_X_'
            direction_BC=0 
            self.Dirichlet_ind[0][cont_n],self.Dirichlet_ind[1][cont_n],\
            self.Dirichlet_ind[2][cont_n]=self.list_BC(read_mesh,name_group,config_mesh,direction_BC,BC_type)
            cont_n+=1
            cont+=1
        if 'BC_Dirichlet_Y_' in config_mesh:
            name_group='BC_Dirichlet_Y_'
            direction_BC=1
            self.Dirichlet_ind[0][cont_n],self.Dirichlet_ind[1][cont_n],\
            self.Dirichlet_ind[2][cont_n]=self.list_BC(read_mesh,name_group,config_mesh,direction_BC,BC_type)
            cont_n+=1
            cont+=1
        if 'BC_Dirichlet_Z_' in config_mesh:
            name_group='BC_Dirichlet_Z_'
            direction_BC=2
            self.Dirichlet_ind[0][cont_n],self.Dirichlet_ind[1][cont_n],\
            self.Dirichlet_ind[2][cont_n]=self.list_BC(read_mesh,name_group,config_mesh,direction_BC,BC_type)
            cont_n+=1
            cont+=1        
        
        self.Dirichlet_ind[0]=np.concatenate(self.Dirichlet_ind[0]) 
        self.Dirichlet_ind[1]=np.concatenate(self.Dirichlet_ind[1])
        self.Dirichlet_ind[2]=np.concatenate(self.Dirichlet_ind[2])            
        
        if cont==0:
            sys.exit('Fatal error: No were informed the Boundary Conditions or\
                     the names not check')
        #Saving number of BC groups
        self.n_BC=cont

#-----------------------------------------------------------------------------     
    def get_number_BC_directions(self,BC,BC_type):
        if BC_type=='Neumman_point':
            name_BC='BC_Neumann_point_K_'

        if BC_type=='Dirichlet_ind':
            name_BC='BC_Dirichlet_K_'
        
        K=['X','Y','Z']
        n_directions=0
        for i in K:
            if name_BC.replace('K', i) in BC:
                n_directions+=1
            
        return n_directions
            
    
    def list_BC(self,read_mesh,name_group,config_mesh,direction_BC,BC_type):
        """ 
        Creates a list cointaning the nodal vector of BC and it's value.
        
        The first position of the list are the are the nodes of each group
        The second position of the list are the values of each group.
        The third position contains vectors with the degrees of freedom of each group
        
        direction_BC - Direction of the BC
        0 - for x 
        1 - for y 
        2 - for z 
        """
        
        n_BC_of_group=len(config_mesh[name_group]) 
        BC_nodes=[[None]*n_BC_of_group]
        BC_values=[[None]*n_BC_of_group]
        BC_DOF=[[None]*n_BC_of_group]
        cont=0;
        flag=0
        
        n_values=len(config_mesh[name_group])
        while flag==0:
            #nodes of the BC group
            name_BC=name_group+str(cont)
            if n_BC_of_group<(cont+1):
                msg1=('Fatal error: '+'error in '+name_group)
                msg2=('. Number of numeric values in config_mesh['+name_group)
                msg3= ('] do not coincide with the BC. Check the number of\
                      groups and the number of numerical values entered.')
                msg=msg1+msg2+msg3
                sys.exit(msg)
            if name_BC not in read_mesh.point_sets:
                msg='Falal error: '+name_BC+' is not defined in mesh file or\
                     name does not confer '
                sys.exit(msg)  
            #Storing nodes of the group
            BC_nodes[cont]=read_mesh.point_sets[name_BC]
            
            #BC group values
            number_of_nodes_group=BC_nodes[cont].shape[0]
            
            if BC_type=='Neumman_point':
                value=config_mesh[name_group][cont]/number_of_nodes_group
                BC_values[cont]=np.ones(number_of_nodes_group)*value
            else:
                BC_values[cont]=np.ones(number_of_nodes_group)*config_mesh[name_group][cont]
            
            #Storing degrees of freedom of the group
            BC_DOF[cont]=np.zeros(number_of_nodes_group,dtype=int)
            cont2=0
            for i in read_mesh.point_sets[name_BC]:
                GDL=i*self.n_GDL_node_element+direction_BC
                BC_DOF[cont][cont2]=GDL
                cont2+=1
                
            if (name_group+str(cont+1)) in read_mesh.point_sets:
                flag=0
                cont+=1
            else:
                flag=-1
                
        BC_nodes=np.concatenate(BC_nodes)
        BC_values=np.concatenate(BC_values)
        BC_DOF=np.concatenate(BC_DOF)
       
        n_BC=cont+1;
        if n_values != n_BC:
            msg1=('Fatal error: '+'error in '+name_group)
            msg2=('. Number of numeric values in config_mesh['+name_group)
            msg3= ('] do not coincide with the BC. Check the number of\
                   groups and the number of numerical values entered.')
            msg=msg1+msg2+msg3
            sys.exit(msg)
        return BC_nodes,BC_values,BC_DOF         
          
         
#-----------------------------------------------------------------------------
    def get_connectivity(self,read_mesh):
        """
        Call methods to allocate connectivity and nodes coordi
        """
        if not (i=="quad" or i=="line" for i in read_mesh.cells_dict):
            sys.exit("Fatal error: Element not implemented. Only C3D8 \
                        hexaedron implemented")
            #Descomentar e colocar essa parte quando implementar hexa.
            #not (i=="quad" or i=="hexahedron" or i=="line" 
            #                            for i in read_mesh.cells_dict):
            #Only one el
        cont=0
        if self.analysis_dimension=='3D':
            if "hexahedron" in read_mesh.cells_dict:
                if "tetra" in read_mesh.cells_dict:
                   sys.exit('Code supports only one element type per mesh.') 
                   
                self.connectivity=read_mesh.cells_dict['hexahedron']
                cont=len(read_mesh.cells_dict['hexahedron'])  
                self.n_nodes_element=8
                self.n_GDL_node_element=3
                self.n_GDL_el=self.n_nodes_element*self.n_GDL_node_element
            if "tetra" in read_mesh.cells_dict:
                sys.exit('Tetrahedral element not implemented.')
                self.connectivity=read_mesh.cells_dict['tetra']
                cont=len(read_mesh.cells_dict['tetra'])
                self.n_nodes_element=4
                self.n_GDL_node_element=3

        elif self.analysis_dimension=='2D_plane_stress':
            sys.exit('Type of analysis not implemented')
            if "quad" in read_mesh.cells_dict: 
                if "triangle" in read_mesh.cells_dict:
                    sys.exit('Code supports only one element type per mesh.')
                    
                self.connectivity=read_mesh.cells_dict['quad']
                cont=len(read_mesh.cells_dict['quad'])
                self.n_nodes_element=4
                self.n_GDL_node_element=2
        
        self.n_elements=cont
        self.n_GDL_tot=self.n_nodes_glob*self.n_GDL_node_element
        
#-----------------------------------------------------------------------------                    


            