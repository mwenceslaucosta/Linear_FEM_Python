# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:03:20 2020

@author: mathe
"""
import numpy as np 
import os 
from mesh import MeshFEM 
import linear_elasticity_iso_3D 
from solvers import  static_linear

class Test_FEM_Class:
    """
    Class to test FEM code according mesh file "cube2.inp" and 
    "displacement_cube1000.inp".
    Linear Elasticity E=210E3 N/mm^2 and poisson=0.29
    Normal force = 200 N
    """

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
        #Config Material model  
        mat_prop=np.array([210E3,0.29])
        material_model=linear_elasticity_iso_3D
        
        #Config Mesh
        config_mesh={}
        config_mesh['mesh_file_name']=os.path.join(mesh_name)
        config_mesh['BC_Neumann_point_X_']=np.array([200])
        config_mesh['BC_Dirichlet_X_']=np.array([0])
        config_mesh['BC_Dirichlet_Y_']=np.array([0])
        config_mesh['BC_Dirichlet_Z_']=np.array([0])
        config_mesh['analysis_dimension']='3D'
        mesh=MeshFEM(config_mesh)
        
        #Solver and Poss-processing
        displacement,stress,strain=static_linear(mesh,material_model,mat_prop)
        
        return displacement
        

#-----------------------------------------------------------------------------# 
        