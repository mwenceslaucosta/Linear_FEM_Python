# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:24:04 2020

@author: Matheus
"""
from numba import jit
import meshio
import numpy as np
def pos_static_linear(B_all_elem,tang_modu,stress_gauss,
                      strain_gauss,connectivity,n_nodes_elem,n_Gauss_elem,
                      DOF_node_elem,displacement,u_elem,n_elem,material_model,
                      stress_nodes,strain_nodes,cont_average,extrapol_vec_stress,
                      extrapol_vec_strain,DOF_stress_strain,extrapol_matrix):
   
    #Stress and strain in gauss points 
    stress_gauss,strain_gauss=material_model.get_stress_and_strain(B_all_elem,
              tang_modu,stress_gauss,strain_gauss,connectivity,n_nodes_elem,
                      n_Gauss_elem,DOF_node_elem,displacement,u_elem,n_elem)
   
    #Stress and strain extrapolated 
    stress_nodes,strain_nodes=extrapolate_stress_strain(stress_gauss,strain_gauss,
                        stress_nodes,strain_nodes,cont_average,connectivity,
                      extrapol_vec_stress,extrapol_vec_strain,DOF_stress_strain,
                        n_Gauss_elem,extrapol_matrix)

    
    return stress_gauss,strain_gauss,stress_nodes,strain_nodes
    
#-----------------------------------------------------------------------------
@jit(nopython=True,cache=True)   
def extrapolate_stress_strain(stress_gauss,strain_gauss,
                         stress_nodes,strain_nodes,cont_average,connectivity,
                        extrapol_vec_stress,extrapol_vec_strain,DOF_stress_strain,
                        n_Gauss_elem,extrapol_matrix):

    for M in range(connectivity.shape[0]):       
        for N in range(DOF_stress_strain):
            extrapol_vec_strain[::1]=0
            extrapol_vec_stress[::1]=0
            for j in range(n_Gauss_elem):
                extrapol_vec_stress[j]=stress_gauss[M,j*DOF_stress_strain+N]
                extrapol_vec_strain[j]=strain_gauss[M,j*DOF_stress_strain+N]
            stress_extra=extrapol_matrix @ extrapol_vec_stress
            strain_extra=extrapol_matrix @ extrapol_vec_strain
            
            for n_node in range(n_Gauss_elem):
                node=connectivity[M,n_node]
                stress_nodes[node,N]=stress_extra[n_node]+stress_nodes[node,N]
                strain_nodes[node,N]=strain_extra[n_node]+strain_nodes[node,N]
                cont_average[node]=cont_average[node]+1
    cont_average=cont_average/DOF_stress_strain
    for i in range(stress_nodes.shape[0]):
        stress_nodes[i,:]=stress_nodes[i,:]/cont_average[i]
        strain_nodes[i,:]=strain_nodes[i,:]/cont_average[i]
    return stress_nodes,strain_nodes
#-----------------------------------------------------------------------------                

def save_results(mesh,displacement,stress_nodes,strain_nodes,out_file_name):
    
    if mesh.DOF_node_elem==2:
        mesh.meshio_.point_data['stress_nodes_X']=stress_nodes[:,0]   
        mesh.meshio_.point_data['stress_nodes_Y']=stress_nodes[:,1]
        mesh.meshio_.point_data['stress_nodes_XY']=stress_nodes[:,2]
        mesh.meshio_.point_data['displacement_X']=displacement[0::2]
        mesh.meshio_.point_data['displacement_Y']=displacement[1::2]
 
        save_file_name=out_file_name+'.vtk'
        #mesh.meshio_.write(save_file_name)
        points = mesh.nodes
        cells = [("quad", mesh.connectivity)]
        meshio.write_points_cells(
        save_file_name,
        points,
        cells,
        # Optionally provide extra data on points, cells, etc.
        point_data= mesh.meshio_.point_data,
        # cell_data=cell_data,
        # field_data=field_data
        )            

   
    elif mesh.DOF_node_elem==3: 
        mesh.meshio_.point_data['stress_nodes_X']=stress_nodes[:,0]   
        mesh.meshio_.point_data['stress_nodes_Y']=stress_nodes[:,1]
        mesh.meshio_.point_data['stress_nodes_Z']=stress_nodes[:,2]
        mesh.meshio_.point_data['stress_nodes_XY']=stress_nodes[:,5]
        mesh.meshio_.point_data['stress_nodes_XZ']=stress_nodes[:,4]
        mesh.meshio_.point_data['stress_nodes_YZ']=stress_nodes[:,3]
        mesh.meshio_.point_data['displacement_X']=displacement[0::3]
        mesh.meshio_.point_data['displacement_Y']=displacement[1::3]
        mesh.meshio_.point_data['displacement_Z']=displacement[2::3]
        
        save_file_name=out_file_name+'.vtk'
        #mesh.meshio_.write(save_file_name)
        points = mesh.nodes
        cells = [("hexahedron", mesh.connectivity)]
        meshio.write_points_cells(
        save_file_name,
        points,
        cells,
        # Optionally provide extra data on points, cells, etc.
        point_data= mesh.meshio_.point_data,
        # cell_data=cell_data,
        # field_data=field_data
        )            
            
            
                
            
