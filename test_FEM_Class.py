# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:03:20 2020

@author: mathe
"""
import numpy as np 
import os 
from mesh import MeshFEM 
import linear_elasticity_iso_3D 
import plane_stress_lin_elast_iso_2D 
from solvers import  static_linear

class Test_FEM_Class:
    """
    Class to test FEM code according mesh file "cube2.inp", 
    "displacement_cube1000.inp" and "cantielever_2D_100x50_bending" .
 
    """

 #-----------------------------------------------------------------------------# 
        
    def test_displacement_cube_2_elements(self):
        """
        Axial test - 3D hexaendron  Mesh with 2 elements        
        Linear Elasticity E=210E3 N/mm^2 and poisson=0.29
        Normal force = 200 N
        """
        f='test_files'
        displacement_cube2=np.loadtxt(os.path.join(f,'displacement_cube2.csv'),delimiter=',')
        mesh_name=os.path.join('test_files','cube2.inp')
        #Config Mesh
        config_mesh={}
        config_mesh['mesh_file_name']=mesh_name
        config_mesh['BC_Neumann_point_X_']=np.array([200])
        config_mesh['BC_Dirichlet_X_']=np.array([0])
        config_mesh['BC_Dirichlet_Y_']=np.array([0])
        config_mesh['BC_Dirichlet_Z_']=np.array([0])
        config_mesh['analysis_dimension']='3D'
        
        out_file_name='FEM_out_test_cube_2'
        
        #Config Material model  
        material_model=linear_elasticity_iso_3D
        result=self.get_result(material_model,config_mesh,out_file_name)
        
        assert np.allclose(result['displacement'],displacement_cube2)

#-----------------------------------------------------------------------------# 
        
    def test_displacement_cube_1000_elements(self):
        """
        Axial test - 3D hexaendron Mesh with 1000 elements        
        Linear Elasticity E=210E3 N/mm^2 and poisson=0.29
        Normal force = 200 N
        """
        f='test_files'
        displacement_cube1000=np.loadtxt(os.path.join(f,'displacement_cube1000.csv'),delimiter=',')
        mesh_name=os.path.join('test_files','cube1000.inp')
        #Config Mesh
        config_mesh={}
        config_mesh['mesh_file_name']=mesh_name
        config_mesh['BC_Neumann_point_X_']=np.array([200])
        config_mesh['BC_Dirichlet_X_']=np.array([0])
        config_mesh['BC_Dirichlet_Y_']=np.array([0])
        config_mesh['BC_Dirichlet_Z_']=np.array([0])
        config_mesh['analysis_dimension']='3D'
        out_file_name='FEM_out_test_cube_1000'
        #Config Material model  
        material_model=linear_elasticity_iso_3D
        result=self.get_result(material_model,config_mesh,out_file_name)
        
        assert np.allclose(result['displacement'],displacement_cube1000)

#-----------------------------------------------------------------------------# 
        
    def test_displacement_cantielever_2D(self):
        """
        Bending test - 2D Mesh  - quad4       
        Linear Elasticity E=210E3 N/mm^2 and poisson=0.29
        Transversal load = 2000 N
        """
        f='test_files'
        displacement=np.loadtxt(os.path.join(f,'cantielever_2D_displacement.csv'),delimiter=',')
        mesh_name=os.path.join('test_files','cantielever_2D_100x50_bending.med')
        #Config Mesh
        config_mesh={}
        config_mesh['mesh_file_name']=mesh_name
        config_mesh['BC_Neumann_point_Y_']=np.array([2000])
        config_mesh['BC_Dirichlet_X_']=np.array([0])
        config_mesh['BC_Dirichlet_Y_']=np.array([0])
        config_mesh['analysis_dimension']='2D'
        config_mesh['Thickness_Group_']=np.array([20])
        out_file_name='FEM_out_test_bending2D_displacement'
        #Config Material model  
        material_model=plane_stress_lin_elast_iso_2D
        result=self.get_result(material_model,config_mesh,out_file_name)
        
        assert np.allclose(result['displacement'],displacement)

#-----------------------------------------------------------------------------# 
        
    def test_normal_stress_X_cantielever_2D(self):
        """
        Bending test - 2D Mesh - quad4    
        Linear Elasticity E=210E3 N/mm^2 and poisson=0.29
        Transversal load = 2000 N
        """
        f='test_files'
        stress_X=np.loadtxt(os.path.join(f,'cantielever_2D_stress_X.csv'),delimiter=',')
        mesh_name=os.path.join('test_files','cantielever_2D_100x50_bending.med')
        #Config Mesh
        config_mesh={}
        config_mesh['mesh_file_name']=mesh_name
        config_mesh['BC_Neumann_point_Y_']=np.array([2000])
        config_mesh['BC_Dirichlet_X_']=np.array([0])
        config_mesh['BC_Dirichlet_Y_']=np.array([0])
        config_mesh['analysis_dimension']='2D'
        config_mesh['Thickness_Group_']=np.array([20])
        out_file_name='FEM_out_test_bending2D_stress_X'
        #Config Material model  
        material_model=plane_stress_lin_elast_iso_2D
        result=self.get_result(material_model,config_mesh,out_file_name)
        
        assert np.allclose(result['stress_nodes'][:,0],stress_X)

#-----------------------------------------------------------------------------# 
        
    def test_shear_stress_XY_cantielever_2D(self):
        """
        Bending test - 2D Mesh        
        Linear Elasticity E=210E3 N/mm^2 and poisson=0.29
        Transversal load = 2000 N
        """
        f='test_files'
        stress_XY=np.loadtxt(os.path.join(f,'cantielever_2D_stress_XY.csv'),delimiter=',')
        mesh_name=os.path.join('test_files','cantielever_2D_100x50_bending.med')
        #Config Mesh
        config_mesh={}
        config_mesh['mesh_file_name']=mesh_name
        config_mesh['BC_Neumann_point_Y_']=np.array([2000])
        config_mesh['BC_Dirichlet_X_']=np.array([0])
        config_mesh['BC_Dirichlet_Y_']=np.array([0])
        config_mesh['analysis_dimension']='2D'
        config_mesh['Thickness_Group_']=np.array([20])
        out_file_name='FEM_out_test_bending2D_stress_XY'
        #Config Material model  
        material_model=plane_stress_lin_elast_iso_2D
        result=self.get_result(material_model,config_mesh,out_file_name)
        
        assert np.allclose(result['stress_nodes'][:,2],stress_XY)
#-----------------------------------------------------------------------------# 
        
    def get_result(self,material_model,config_mesh,out_file_name):

        mat_prop=np.array([210E3,0.29])
        mesh=MeshFEM(config_mesh)
        
        #Solver and Poss-processing   
        result=static_linear(mesh,material_model,mat_prop,out_file_name)
        return result
        

#-----------------------------------------------------------------------------# 
        