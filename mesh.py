# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:55:50 2020

@author: Matheus_Wenceslau
"""
import numpy as np
import os
import meshio 
import sys 

class MeshFEM:
    """
    Class to import mesh in .med from Salome, and .inp format (Abaqus) from GMSH
    Elements: C3D8-Hexahedron 
              Quad-4 
              Triangle-3
              Tetra-4
     """         
#-----------------------------------------------------------------------------    
    def __init__(self,config_mesh):
        """
        Constructor meshFEM class 
        """
        self.mesh_file=os.path.join(config_mesh['mesh_file_name'])
        self.meshio_=meshio.read(self.mesh_file)    
        self.analysis_dimension=config_mesh['analysis_dimension']        
        self.nodes=self.meshio_.points
        self.n_nodes_glob=self.nodes.shape[0]
 #      self.lines=read_mesh.cells_dict['line']
        
        if config_mesh['mesh_file_name'].endswith('.inp'):
            #Abaqus Format - .inp
            self.mesh_type='Abaqus'
        elif config_mesh['mesh_file_name'].endswith('.med'):
            #Salome Format - .med 
            self.mesh_type='Salome'
        else:
            sys.exit('Fatal error: Mesh format not accepted')
        
        self.get_connectivity(self.meshio_)

        #Boundary Conditions 
        if config_mesh['mesh_file_name'].endswith('.inp'):
            #Abaqus Format - .inp
            self.get_BC_inp(self.meshio_,config_mesh)
        elif config_mesh['mesh_file_name'].endswith('.med'):
            #Salome Format - .med 
            self.get_BC_med(config_mesh)
        
        #Thickness 2D Analysis 
        if config_mesh['analysis_dimension']=='2D':
            if 'Thickness_Group_' not in config_mesh:
                sys.exit('Fatal error: Thickness not informed')
            else:
                if len(config_mesh['Thickness_Group_'])>1:
                    self.get_thickness_groups_med(self.meshio_,config_mesh)
                else:
                    thickness_value=config_mesh['Thickness_Group_']
                    self.thickness_vector=np.ones(self.n_elem)*thickness_value
                      
#-----------------------------------------------------------------------------
            
    def get_BC_med (self,config_mesh):
        """
        Method to get the list of nodes and values of the Boundary Conditions(BC)
        in .med format (salome).
        
        direction_BC - Direction of the BC
        0 - for x 
        1 - for y 
        2 - for z 
        
        self.Neumann_pt_nodes: Vector containing all nodes with nodal Neumman BC 
        self.Neumann_pt_values: Vector containing BC value of each DOF
        self.Neumann_pt_DOF: Vector containing nodal Neumman BC DOF.
        
        self.Dirichlet_nodes: Vector containing all nodes with Dirichlet BC 
        self.Dirichlet_values: Vector containing BC value of each DOF
        self.Dirichlet_DOF: Vector containing Dirichlet BC DOF.
        """
        BC_index_array=(np.argwhere(self.meshio_.point_data['point_tags']>1)).reshape(-1)
        self.Neumann_pt_nodes=[]
        self.Neumann_pt_values=[]
        self.Neumann_pt_DOF=[]
        self.Dirichlet_nodes=[]
        self.Dirichlet_values=[]
        self.Dirichlet_DOF=[]
        for index_BC in BC_index_array:
            node=index_BC
            BC_key=self.meshio_.point_data['point_tags'][index_BC]
            for ii in self.meshio_.point_tags[BC_key]:
                if ii != 'Group_Of_All_Nodes':
                    self.BC_med(config_mesh,ii,node,BC_key)
        self.Neumann_pt_nodes=np.asarray(self.Neumann_pt_nodes)
        self.Neumann_pt_values=np.asarray(self.Neumann_pt_values)
        self.Neumann_pt_DOF=np.asarray(self.Neumann_pt_DOF)
        self.Dirichlet_nodes=np.asarray(self.Dirichlet_nodes)
        self.Dirichlet_values=np.asarray(self.Dirichlet_values)
        self.Dirichlet_DOF=(np.asarray(self.Dirichlet_DOF)).reshape(-1)
        self.Dirichlet_DOF_sorted=np.sort(self.Dirichlet_DOF)

            
#-----------------------------------------------------------------------------
    def BC_med(self,config_mesh,name_BC,node,BC_key):
        """
        Method to call routine list_BC_med in .med format (salome)
        
        direction_BC - Direction of the BC
        0 - for x 
        1 - for y 
        2 - for z 
        """
        
        suffix=int(name_BC[-1])
        
        if name_BC.startswith('BC_Neumann_point_X_'):
            name_prefix='BC_Neumann_point_X_'
            direction_BC=0
            self.list_BC_med(config_mesh,name_prefix,suffix,direction_BC,node,BC_key)
        elif name_BC.startswith('BC_Neumann_point_Y_'):
            name_prefix='BC_Neumann_point_Y_'
            direction_BC=1
            self.list_BC_med(config_mesh,name_prefix,suffix,direction_BC,node,BC_key)
        elif name_BC.startswith('BC_Neumann_point_Z_'):
            name_prefix='BC_Neumann_point_Z_'
            direction_BC=2
            self.list_BC_med(config_mesh,name_prefix,suffix,direction_BC,node,BC_key)
        
        elif name_BC.startswith('BC_Dirichlet_X_'):
            name_prefix='BC_Dirichlet_X_'
            direction_BC=0
            self.list_BC_med(config_mesh,name_prefix,suffix,direction_BC,node,BC_key)
        elif name_BC.startswith('BC_Dirichlet_Y_'):
            name_prefix='BC_Dirichlet_Y_'
            direction_BC=1
            self.list_BC_med(config_mesh,name_prefix,suffix,direction_BC,node,BC_key)
        elif name_BC.startswith('BC_Dirichlet_Z_'):
            name_prefix='BC_Dirichlet_Z_'
            direction_BC=2
            self.list_BC_med(config_mesh,name_prefix,suffix,direction_BC,node,BC_key)
        else: 
            msg='Falal error: '+name_BC+' is not defined in mesh file or\
                     or the names does not match. '
            sys.exit(msg)
            
#-----------------------------------------------------------------------------
    def list_BC_med(self,config_mesh,name_prefix,suffix,direction_BC,node,BC_key):
        """ 
        Method to create a list cointaning the nodal vector of BC and it's value in 
        .med (Salome) format.
                 
        """
        
        name_group=name_prefix+str(suffix)
        if len(config_mesh[name_prefix]) != suffix+1:
            msg1=('Fatal error: '+'error in '+name_group)
            msg2=('. Number of numeric values in config_mesh['+name_group)
            msg3= ('] does not coincide with the BC. Check the number of\
                          groups and the number of numerical values informed.')
            msg=msg1+msg2+msg3
            sys.exit(msg)
            
        DOF=self.DOF_node_elem*node+direction_BC
        if name_prefix.startswith('BC_Neumann'):
            unique,counts=np.unique(self.meshio_.point_data['point_tags'],return_counts=True) 
            number_of_nodes_group=counts[BC_key-1]
            self.Neumann_pt_nodes.append(node)
            self.Neumann_pt_values.append(config_mesh[name_prefix][suffix]/number_of_nodes_group) 
            self.Neumann_pt_DOF.append(DOF)
        elif name_prefix.startswith('BC_Dirichlet'):
            self.Dirichlet_nodes.append(node)
            self.Dirichlet_values.append(config_mesh[name_prefix][suffix]) 
            self.Dirichlet_DOF.append(DOF)
        
        
 #-----------------------------------------------------------------------------                                                  
                           
    def get_BC_inp(self,read_mesh,config_mesh):
        """
        Method to get the list of nodes and values of the Boundary Conditions(BC)
        in .inp format (Abaqus from GMSH)
        
        direction_BC - Direction of the BC
        0 - for x 
        1 - for y 
        2 - for z 
        
        self.Neumann_pt_nodes: Vector containing all nodes with nodal Neumman BC. 
        self.Neumann_pt_values: Vector containing BC value of each DOF.
        self.Neumann_pt_DOF: Vector containing nodal Neumman BC DOF.
        
        self.Dirichlet_nodes: Vector containing all nodes with Dirichlet BC 
        self.Dirichlet_values: Vector containing BC value of each DOF
        self.Dirichlet_DOF: Vector containing Dirichlet BC DOF.
        """
        cont=0
        #Neumann nodal
        Neumman=['BC_Neumann_point_X_','BC_Neumann_point_Y_','BC_Neumann_point_Z_']
        if any(s==Neumman[0] or s==Neumman[1] or s==Neumman[2] for s in config_mesh):
            BC_type='Neumman_point'
            n=self.get_number_BC_directions(config_mesh,BC_type)
            Neumann_pt=[[None]*n,[None]*n,[None]*n]
            cont_n=0
            if 'BC_Neumann_point_X_' in config_mesh:
                name_group='BC_Neumann_point_X_'
                direction_BC=0 
                Neumann_pt[0][cont_n],Neumann_pt[1][cont_n],\
                Neumann_pt[2][cont_n]=self.list_BC_inp(read_mesh,name_group,config_mesh,direction_BC,BC_type)
                cont_n+=1
                cont+=1
            if 'BC_Neumann_point_Y_' in config_mesh:
                name_group='BC_Neumann_point_Y_'
                direction_BC=1
                Neumann_pt[0][cont_n],Neumann_pt[1][cont_n],\
                Neumann_pt[2][cont_n]=self.list_BC_inp(read_mesh,name_group,config_mesh,direction_BC,BC_type)
                cont_n+=1
                cont+=1
            if 'BC_Neumann_point_Z_' in config_mesh:
                name_group='BC_Neumann_point_Z_'
                direction_BC=2
                Neumann_pt[0][cont_n],Neumann_pt[1][cont_n],\
                Neumann_pt[2][cont_n]=self.list_BC_inp(read_mesh,name_group,config_mesh,direction_BC,BC_type)
                cont_n+=1
                cont+=1
                
            self.Neumann_pt_nodes=np.concatenate(Neumann_pt[0]).reshape(-1)
            self.Neumann_pt_values=np.concatenate(Neumann_pt[1]).reshape(-1)
            self.Neumann_pt_DOF=np.concatenate(Neumann_pt[2]).reshape(-1)

        
        #Dirichlet
        Dirichlet=['BC_Dirichlet_X_','BC_Dirichlet_Y_','BC_Dirichlet_Z_']
        if any(s==Dirichlet[0] or s==Dirichlet[1] or s==Dirichlet[2] for s in config_mesh):
            BC_type='Dirichlet_ind'
            n=self.get_number_BC_directions(config_mesh,BC_type)
            Dirichlet_ind=[[None]*n,[None]*n,[None]*n]
            cont_n=0
            if 'BC_Dirichlet_X_' in config_mesh:
                name_group='BC_Dirichlet_X_'
                direction_BC=0 
                Dirichlet_ind[0][cont_n],Dirichlet_ind[1][cont_n],\
                Dirichlet_ind[2][cont_n]=self.list_BC_inp(read_mesh,name_group,config_mesh,direction_BC,BC_type)
                cont_n+=1
                cont+=1
            if 'BC_Dirichlet_Y_' in config_mesh:
                name_group='BC_Dirichlet_Y_'
                direction_BC=1
                Dirichlet_ind[0][cont_n],Dirichlet_ind[1][cont_n],\
                Dirichlet_ind[2][cont_n]=self.list_BC_inp(read_mesh,name_group,config_mesh,direction_BC,BC_type)
                cont_n+=1
                cont+=1
            if 'BC_Dirichlet_Z_' in config_mesh:
                name_group='BC_Dirichlet_Z_'
                direction_BC=2
                Dirichlet_ind[0][cont_n],Dirichlet_ind[1][cont_n],\
                Dirichlet_ind[2][cont_n]=self.list_BC_inp(read_mesh,name_group,config_mesh,direction_BC,BC_type)
                cont_n+=1
                cont+=1        
            
            self.Dirichlet_nodes=np.concatenate(Dirichlet_ind[0]).reshape(-1) 
            self.Dirichlet_values=np.concatenate(Dirichlet_ind[1]).reshape(-1)
            self.Dirichlet_DOF=np.concatenate(Dirichlet_ind[2]).reshape(-1)
            self.Dirichlet_DOF_sorted=np.sort(self.Dirichlet_DOF)
        else:
            sys.exit('Fatal error: No were informed the Dirichlet Boundary\
                     conditions or the names does not match.')

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
            
    
    def list_BC_inp(self,read_mesh,name_group,config_mesh,direction_BC,BC_type):
        """ 
        Creates a list cointaning the nodal vector of BC and it's value for 
        abaqus format.
        
        The first position of the list are the nodes of each group.
        The second position of the list are the values of each group.
        The third position contains vectors with the degrees of freedom of each group.
        
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
                msg3= ('] does not coincide with the BC. Check the number of\
                      groups and the number of numerical values entered.')
                msg=msg1+msg2+msg3
                sys.exit(msg)
            if name_BC not in read_mesh.point_sets:
                msg='Falal error: '+name_BC+' is not defined in mesh file or\
                     the name does not confer '
                sys.exit(msg)  
            #Storing nodes of the group
            BC_nodes[0][cont]=read_mesh.point_sets[name_BC]
            
            #BC group values
            number_of_nodes_group=BC_nodes[0][cont].shape[0]
            
            if BC_type=='Neumman_point':
                value=config_mesh[name_group][cont]/number_of_nodes_group
                BC_values[0][cont]=np.ones(number_of_nodes_group)*value
            else:
                BC_values[0][cont]=np.ones(number_of_nodes_group)*config_mesh[name_group][cont]
            
            #Storing degrees of freedom of the group
            BC_DOF[0][cont]=np.zeros(number_of_nodes_group,dtype=int)
            cont2=0
            for i in read_mesh.point_sets[name_BC]:
                DOF=i*self.DOF_node_elem+direction_BC
                BC_DOF[0][cont][cont2]=DOF
                cont2+=1
                
            if (name_group+str(cont+1)) in read_mesh.point_sets:
                flag=0
                cont+=1
            else:
                flag=-1
                
        BC_nodes=np.concatenate(BC_nodes).reshape(-1)
        BC_values=np.concatenate(BC_values).reshape(-1)
        BC_DOF=np.concatenate(BC_DOF).reshape(-1)
       
        n_BC=cont+1;
        if n_values != n_BC:
            msg1=('Fatal error: '+'error in '+name_group)
            msg2=('. Number of numeric values in config_mesh['+name_group)
            msg3= ('] does not coincide with the BC. Check the number of\
                   groups and the number of numerical values entered.')
            msg=msg1+msg2+msg3
            sys.exit(msg)
        return BC_nodes,BC_values,BC_DOF         
          
         
#-----------------------------------------------------------------------------

    def get_thickness_groups_med(self,read_mesh,config_mesh):
        """ 
        Method to get a vector containing the thickness of all elements
        in med format
        """
        n_thickness=len(config_mesh['Thickness_Group_'])
        self.thickness_vector=np.zeros(self.n_elem)
        for elem in range(self.n_elem):
            thickness_key=read_mesh.cell_data['cell_tags'][0][elem]
            for key in read_mesh.cell_tags[thickness_key]:
                if key!='Group_Of_All_Faces':
                    suffix=int(key[-1])
                    if n_thickness<suffix+1:
                        sys.exit('Fatal error. Number of thickness groups does \
                                 not match with the mesh thcickness number')
                    else:
                        thickness_value=config_mesh['Thickness_Group_'][suffix]
                        self.thickness_vector[elem]=thickness_value
            

#-----------------------------------------------------------------------------
    def get_connectivity(self,read_mesh):
        """
        Call methods to allocate connectivity and nodes coord
        """
        cont=0
        if self.analysis_dimension=='3D':
            if "hexahedron" in read_mesh.cells_dict:
                #Hexa8
                if "tetra" in read_mesh.cells_dict:
                   sys.exit('Code supports only one element type per mesh.') 
                   
                self.connectivity=read_mesh.cells_dict['hexahedron']
                cont=len(read_mesh.cells_dict['hexahedron'])  
                self.n_nodes_elem=8
                self.DOF_node_elem=3
                self.n_Gauss_elem=8
                self.DOF_stress_strain=6
                self.DOF_elem=self.n_nodes_elem*self.DOF_node_elem
                self.element_name="hexahedron"

                import hexaedron_8nodes 
                self.fun_elem=hexaedron_8nodes 
                   
            if "tetra" in read_mesh.cells_dict:
                if self.mesh_type=='Abaqus':
                    sys.exit('Fatal error: Tetra element implemented only\
                             for Salome mesh type')
                
                #Tetra4
                self.connectivity=read_mesh.cells_dict['tetra']
                cont=len(read_mesh.cells_dict['tetra'])
                self.n_nodes_elem=4
                self.DOF_node_elem=3
                self.n_Gauss_elem=1
                self.DOF_stress_strain=6
                self.DOF_elem=self.n_nodes_elem*self.DOF_node_elem
                self.element_name="tetra"
                import tetrahedron_4nodes 
                self.fun_elem=tetrahedron_4nodes

        elif self.analysis_dimension=='2D':          
            if "quad" in read_mesh.cells_dict: 
                #Quad4
                if "triangle" in read_mesh.cells_dict:
                    sys.exit('Code supports only one element type per mesh.')
                    
                self.connectivity=read_mesh.cells_dict['quad']
                cont=len(read_mesh.cells_dict['quad'])
                self.n_nodes_elem=4
                self.DOF_node_elem=2
                self.n_Gauss_elem=4
                self.DOF_stress_strain=3
                self.DOF_elem=self.n_nodes_elem*self.DOF_node_elem
                self.element_name="quad"
                import quad_4nodes
                self.fun_elem=quad_4nodes

            if "triangle" in read_mesh.cells_dict:
                #Triangle3

                self.connectivity=read_mesh.cells_dict['triangle']
                cont=len(read_mesh.cells_dict['triangle'])
                self.n_nodes_elem=3
                self.DOF_node_elem=2
                self.n_Gauss_elem=1
                self.DOF_stress_strain=3
                self.DOF_elem=self.n_nodes_elem*self.DOF_node_elem
                self.element_name="triangle"
                import triangle_3nodes
                self.fun_elem=triangle_3nodes
     
        self.n_elem=cont
        self.DOF_tot=self.n_nodes_glob*self.DOF_node_elem
        
#-----------------------------------------------------------------------------                    


            
