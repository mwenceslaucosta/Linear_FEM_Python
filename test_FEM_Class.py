# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:03:20 2020

@author: mathe
"""
import numpy as np 
from scipy.sparse import linalg
import os 
from mesh import MeshFEM 
from hexahedron_8nodes import Hexaedron_8nodes
from inload import Inload
from constitutive_models import linear_elasticity_iso_3D
from assembly_KGlob_and_BC import Assembly_KGlob_and_BC

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
        
    def test_displacement_cube_2_elements(self):
        f='test_files'
        displacement_cube2=np.loadtxt(os.path.join(f,'displacement_cube2.csv'),delimiter=',')
        mesh_name=os.path.join('test_files','cube2.inp')
        displacement_computed=self.get_displacement(mesh_name)
        
        assert np.allclose(displacement_computed,displacement_cube2)

#-----------------------------------------------------------------------------# 
        
    def test_displacement_cube_1000_elements(self):
        f='test_files'
        displacement_cube1000=np.loadtxt(os.path.join(f,'displacement_cube1000.csv'),delimiter=',')
        mesh_name=os.path.join('test_files','cube1000.inp')
        displacement_computed=self.get_displacement(mesh_name)
        
        assert np.allclose(displacement_computed,displacement_cube1000)

#-----------------------------------------------------------------------------# 
        
    def get_displacement(self,mesh_name):
        config_mesh={}
        config_mesh['mesh_file_name']=mesh_name
        config_mesh['BC_Neumann_point_X_']=np.array([200])
        config_mesh['BC_Dirichlet_X_']=np.array([0])
        config_mesh['BC_Dirichlet_Y_']=np.array([0])
        config_mesh['BC_Dirichlet_Z_']=np.array([0])
        config_mesh['analysis_dimension']='3D'
        mesh=MeshFEM(config_mesh)
        external_force=Inload(mesh)
        
        #Material properties 
        config_material=np.array([210E3,0.29])
        
        #Elementar stifiness 
        element=[None]*mesh.n_elements
        material=[None]*mesh.n_elements
           
        for i in range(mesh.n_elements):
            element[i]=Hexaedron_8nodes(mesh,i)
            material[i]=linear_elasticity_iso_3D(element[i],config_material)        
            element[i].jacobian_element()    
            element[i].B_element() 
            element[i].get_Ke_element(material[i]) 
        
        #Global stifiness     
        K_Glob=Assembly_KGlob_and_BC(mesh)
        K_Glob.KGlobal(element,mesh,material,external_force.load_vector)

        #Displacement
        displacement=linalg.spsolve(K_Glob.KGlob_csc_BC,external_force.load_vector)
        
        return displacement
        

#-----------------------------------------------------------------------------# 
        