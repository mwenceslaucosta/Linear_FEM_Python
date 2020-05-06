# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:04:18 2020

@author: mathe
"""

from mesh import MeshFEM 
import numpy as np
import meshio 
from hexahedron_8nodes import Hexaedron_8nodes
config_mesh={}
config_mesh['mesh_file_name']='cubo.inp'
config_mesh['BC_Neumann_X_']=np.array([0])
config_mesh['BC_Neumann_Z_']=np.array([0])
config_mesh['BC_Neumann_Y_']=np.array([0])
config_mesh['analysis_dimension']='3D'
mesh=MeshFEM(config_mesh)



cd_node=np.array(([0,0,0],[4,0,0],[4,1,0],[0,1,0],[0,0,2],[4,0,2],[4,1,2],[0,1,2]))
element=Hexaedron_8nodes()
element.jacobian_element(cd_node)
element.B_element()