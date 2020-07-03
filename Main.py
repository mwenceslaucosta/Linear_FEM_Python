# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:56:49 2020

@author: Matheus E. Wenceslau Costa
"""
import numpy as np 
from mesh import MeshFEM 
import linear_elasticity_iso_3D 
from solvers import static_linear


#Config Mesh
config_mesh={}
config_mesh['mesh_file_name']='cube1000.inp' #'test_mesh_linux.med'
config_mesh['BC_Neumann_point_X_']=np.array([200])
config_mesh['BC_Dirichlet_X_']=np.array([0])
config_mesh['BC_Dirichlet_Y_']=np.array([0])
config_mesh['BC_Dirichlet_Z_']=np.array([0])
config_mesh['analysis_dimension']='3D'
mesh=MeshFEM(config_mesh)

#Config Material model  
# mat_prop[0] --> elastic modulus - E
# mat_prop[1] --> poisson - nu

material_model=linear_elasticity_iso_3D
mat_prop=np.array([210E3,0.29])

#Out file name 
out_file_name='FEM_out'

#Solver and Poss-processing
displacement,stress_gauss,strain_gauss,stress_nodes,strain_nodes=static_linear(
                                    mesh,material_model,mat_prop,out_file_name)



