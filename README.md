# FEM_Python

* Código FEM para análise linear.
* Pré processamento: Pode ser realizada no Salome ou Gmsh. 
* Pós-processamento: O arquivo de saída gerado pode ser utilizado no Paraview.

## Etapas para configuração da Análise 

1.	Primeiro deve ser informado o nome do arquivo contendo a malha. Atualmente está implementado elemento 3D hexaédrico e 2D quadrangular. Os arquivos devem estar no formato .med (Salome) ou no formato .inp (Gmsh-Abaqus). A configuração do nome do arquivo é feita no dicionário config_mesh, com a chave “mesh_file_name”, conforme exemplo abaixo. 

		Exemplo: config_mesh['mesh_file_name']='cube1000_bending.inp' 

2.	 Informar valores das condições de contorno (BC) para cada grupo de condição de contorno criado no gerador de malha. O código está preparado para receber valores de deslocamento (condições de Dirichlet) e de força nodal pontual (condições de Neumann) nas direções X, Y e Z. Os vetores seguem o seguinte padrão: 
	
	Configuração para BC do tipo Neumann nodal nas direções X, Y e Z:
		config_mesh['BC_Neumann_point_X_']         
		config_mesh['BC_Neumann_point_Y_']           
		config_mesh[‘BC_Neumann_point_Z_’]
	
	Configuração para BC do tipo Dirichlet nodal nas direções X, Y e Z:
		config_mesh['BC_Dirichlet_X_']             
		config_mesh['BC_Dirichlet _Z_']            
		config_mesh['BC_Dirichlet_Y_']

	Os valores devem ser fornecidos em um vetor que relaciona o nome da condição de contorno com o respectivo 
	grupo criado. Por exemplo, em um caso onde foram criados dois grupos do tipo “BC_Neumann_point_X_”, onde o 
	primeiro  grupo tem valor aplicado de 100N e o segundo 200N, o vetor informado deve ser da seguinte forma: 
	
		config_mesh['BC_Neumann_point_X_']=np.array([100, 200])
	
	Observe que o “BC_Neumann_point_X_” carrega os valores de todos os grupos do tipo BC_Neumann_point_X_.

	Atenção: A nomeação dos grupos no gerador de malha deve seguir o seguinte padrão: 
    
		Grupo 1 do tipo Neumann pontual: BC_Neumann_point_X_0 
		Grupo 2 do tipo Neumann pontual: BC_Neumann_point_X_1.
		Continua até o grupo N desejado

		Grupo 1 do tipo Dirichlet: BC_Dirichlet_X_0 
		Grupo 2 do tipo Dirichlet: BC_Dirichlet_X_1. 
		Continua até o grupo N desejado

3.	Informar tipo de elemento (3D ou plane stress): 
		
		Obs: Somente 3D implementado no momento. 
		
		Exemplo: config_mesh['analysis_dimension']='3D'

4.	Informar nome do arquivo de saída: 
		
		Exemplo: out_file_name='FEM_out'
		
		Obs: Por default os dados são salvos no formato .vtk. Este formato pode ser lido facilmente no paraview. 

5.	Configurar o modelo constitutivo 
	
	5.a	Informar o modelo constitutivo: 
        
		Obs: Implementado somente linear elástico 3D por enquanto. 
        
		material_model=linear_elasticity_iso_3D

	5.b	Informar parâmetros materiais elásticos: Módulo elástico e coeficiente de Poisson no vetor numpy mat_prop. A primeira posição é o módulo de elasticidade, a segunda corresponde ao Poisson.
      
	     mat_prop=np.array([210E3,0.29])

6.	Informar nome do arquivo de saída.
		Exemplo: out_file_name='FEM_out'
		Obs: Por default os dados são salvos no formato .vtk. Este formato pode ser lido facilmente no paraview. 
