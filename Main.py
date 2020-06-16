# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:56:49 2020

@author: mathe
"""
import numpy as np 
import os 
from mesh import MeshFEM 
import linear_elasticity_iso_3D 
from solvers import  static_linear
from test_FEM_Class import Test_FEM_Class

#Config Material model  
# mat_prop[0] --> elastic modulus - E
# mat_prop[1] --> poisson - nu
mat_prop=np.array([210E3,0.29])
material_model=linear_elasticity_iso_3D

#Config Mesh
config_mesh={}
config_mesh['mesh_file_name']=os.path.join('cube1000.inp')
config_mesh['BC_Neumann_point_X_']=np.array([200])
#config_mesh['BC_Dirichlet_X_']=np.array([0, 4.7619E-4])
config_mesh['BC_Dirichlet_X_']=np.array([0])
config_mesh['BC_Dirichlet_Y_']=np.array([0])
config_mesh['BC_Dirichlet_Z_']=np.array([0])
config_mesh['analysis_dimension']='3D'
mesh=MeshFEM(config_mesh)

#Solver and Poss-processing
displacement,stress,strain=static_linear(mesh,material_model,mat_prop)
