# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:03:20 2020

@author: mathe
"""
import numpy as np 
import os 
from mesh import MeshFEM 
from hexahedron_8nodes import Hexaedron_8nodes
from inload import Inload
from constitutive_models import linear_elasticity_iso_3D
from assembly_KGlob import KGlobal
from dirichlet_bc_imposition import Dirichlet_imposition

class Test_FEM_Class:
    """
    Class to test FEM code according mesh file "cube2.inp".
    Linear Elasticity E=210E3 N/mm^2 and poisson=0.29
    """

#-----------------------------------------------------------------------------# 

    def test_elementary_stiffness_0(self):
        f='test_files'
        self.Ke_0=np.loadtxt(os.path.join(f,'K_element_0_cube2.csv'),delimiter=',')
        config_mesh={}
        config_mesh['mesh_file_name']=os.path.join('test_files','cube2.inp')
        config_mesh['BC_Neumann_point_X_']=np.array([200])
        config_mesh['BC_Dirichlet_X_']=np.array([0])
        config_mesh['BC_Dirichlet_Y_']=np.array([0])
        config_mesh['BC_Dirichlet_Z_']=np.array([0])
        config_mesh['analysis_dimension']='3D'
        mesh=MeshFEM(config_mesh)
        
        #Material properties 
        config_material=np.array([210E3,0.29])
        
        #Elementar stifiness 
        self.element=[None]*mesh.n_elements
        material=[None]*mesh.n_elements
        for i in range(mesh.n_elements):
            self.element[i]=Hexaedron_8nodes(mesh,i)
            material[i]=linear_elasticity_iso_3D(self.element[i],config_material)
        
        for i in range(mesh.n_elements): 
            self.element[i].jacobian_element()    
            self.element[i].B_element() 
        

        for i in range(mesh.n_elements):     
            self.element[i].get_Ke_element(material[i]) 
        
        assert np.allclose(self.element[0].Ke,self.Ke_0)
#-----------------------------------------------------------------------------# 
        
    def test_elementary_stiffness_1(self):
        f='test_files'
        self.Ke_1=np.loadtxt(os.path.join(f,'K_element_1_cube2.csv'),delimiter=',')
        config_mesh={}
        config_mesh['mesh_file_name']=os.path.join('test_files','cube2.inp')
        config_mesh['BC_Neumann_point_X_']=np.array([200])
        config_mesh['BC_Dirichlet_X_']=np.array([0])
        config_mesh['BC_Dirichlet_Y_']=np.array([0])
        config_mesh['BC_Dirichlet_Z_']=np.array([0])
        config_mesh['analysis_dimension']='3D'
        mesh=MeshFEM(config_mesh)
        
        #Material properties 
        config_material=np.array([210E3,0.29])
        
        #Elementar stifiness 
        self.element=[None]*mesh.n_elements
        material=[None]*mesh.n_elements
        for i in range(mesh.n_elements):
            self.element[i]=Hexaedron_8nodes(mesh,i)
            material[i]=linear_elasticity_iso_3D(self.element[i],config_material)
        
        for i in range(mesh.n_elements): 
            self.element[i].jacobian_element()    
            self.element[i].B_element() 
        

        for i in range(mesh.n_elements):     
            self.element[i].get_Ke_element(material[i]) 
        assert np.allclose(self.element[1].Ke,self.Ke_1)
#-----------------------------------------------------------------------------# 
        
    def test_global_stiffness(self):
        f='test_files'
        self.KG=np.loadtxt(os.path.join(f,'KG_cube2.csv'),delimiter=',')
        config_mesh={}
        config_mesh['mesh_file_name']=os.path.join('test_files','cube2.inp')
        config_mesh['BC_Neumann_point_X_']=np.array([200])
        config_mesh['BC_Dirichlet_X_']=np.array([0])
        config_mesh['BC_Dirichlet_Y_']=np.array([0])
        config_mesh['BC_Dirichlet_Z_']=np.array([0])
        config_mesh['analysis_dimension']='3D'
        mesh=MeshFEM(config_mesh)
        
        #Material properties 
        config_material=np.array([210E3,0.29])
        
        #Elementar stifiness 
        self.element=[None]*mesh.n_elements
        material=[None]*mesh.n_elements
        for i in range(mesh.n_elements):
            self.element[i]=Hexaedron_8nodes(mesh,i)
            material[i]=linear_elasticity_iso_3D(self.element[i],config_material)
        
        for i in range(mesh.n_elements): 
            self.element[i].jacobian_element()    
            self.element[i].B_element() 
        

        for i in range(mesh.n_elements):     
            self.element[i].get_Ke_element(material[i]) 
        
        #Global stifiness     
        self.K_glob=np.zeros((mesh.n_GDL_tot,mesh.n_GDL_tot))
        self.K_glob=KGlobal(self.K_glob,self.element,mesh,material)
        assert np.allclose(self.K_glob,self.KG)   
#-----------------------------------------------------------------------------# 
        
    def test_global_stiffness_BC_impoesd(self):
        f='test_files'
        self.KG_BC_imposed=np.loadtxt(os.path.join(f,'KG_BC_imposed_cube2.csv'),delimiter=',')
        config_mesh={}
        config_mesh['mesh_file_name']=os.path.join('test_files','cube2.inp')
        config_mesh['BC_Neumann_point_X_']=np.array([200])
        config_mesh['BC_Dirichlet_X_']=np.array([0])
        config_mesh['BC_Dirichlet_Y_']=np.array([0])
        config_mesh['BC_Dirichlet_Z_']=np.array([0])
        config_mesh['analysis_dimension']='3D'
        mesh=MeshFEM(config_mesh)
        self.external_force=Inload(mesh)
        
        #Material properties 
        config_material=np.array([210E3,0.29])
        
        #Elementar stifiness 
        self.element=[None]*mesh.n_elements
        material=[None]*mesh.n_elements
        for i in range(mesh.n_elements):
            self.element[i]=Hexaedron_8nodes(mesh,i)
            material[i]=linear_elasticity_iso_3D(self.element[i],config_material)
        
        for i in range(mesh.n_elements): 
            self.element[i].jacobian_element()    
            self.element[i].B_element() 
        

        for i in range(mesh.n_elements):     
            self.element[i].get_Ke_element(material[i]) 
        
        #Global stifiness     
        self.K_glob=np.zeros((mesh.n_GDL_tot,mesh.n_GDL_tot))
        self.K_glob=KGlobal(self.K_glob,self.element,mesh,material)
        
        self.K_glob_BC_imposed=np.zeros((mesh.n_GDL_tot,mesh.n_GDL_tot))
        self.external_force_BC=np.zeros(mesh.n_GDL_tot)
        
        self.K_glob_BC_imposed[:,:]=self.K_glob
        #Dirichlet Boundary Condition imposition
        self.external_force_BC[:]=self.external_force.load_vector
        BC_imposition=Dirichlet_imposition(mesh,self.external_force.load_vector)
        BC_imposition.dirichlet_imposition(mesh,self.K_glob_BC_imposed,
                                                       self.external_force_BC)
        assert np.allclose(self.K_glob_BC_imposed,self.KG_BC_imposed)
#-----------------------------------------------------------------------------# 
        
    def test_external_force(self):
        f='test_files'
        self.load=np.loadtxt(os.path.join(f,'load_cube2.csv'),delimiter=',')
        config_mesh={}
        config_mesh['mesh_file_name']=os.path.join('test_files','cube2.inp')
        config_mesh['BC_Neumann_point_X_']=np.array([200])
        config_mesh['BC_Dirichlet_X_']=np.array([0])
        config_mesh['BC_Dirichlet_Y_']=np.array([0])
        config_mesh['BC_Dirichlet_Z_']=np.array([0])
        config_mesh['analysis_dimension']='3D'
        mesh=MeshFEM(config_mesh)
        self.external_force=Inload(mesh)
        
        assert np.allclose(self.external_force.load_vector,self.load)
#-----------------------------------------------------------------------------# 

        