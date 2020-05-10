# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:31:14 2020

@author: mathe
"""
import numpy as np 

class Hexaedron_8nodes:    
    
    """ 
    class to formulate 8 nodes isoparametric hexahedron element. 
    """
    def __init__(self,mesh,element_number):
        self.n_gauss=8
        self.n_nodes_element=8
        self.GDL_no_element=3
        self.element_number=element_number
        self.coordinantes_elem=np.zeros((8,self.GDL_no_element))
        self.get_coordinantes_nodes_elem(mesh)
        self.phi=np.zeros((self.n_nodes_element,self.n_gauss))
        self.jacobian=np.zeros((self.GDL_no_element,self.GDL_no_element)) 
        self.det_Jacobian=np.zeros(self.n_gauss)
        self.B=np.zeros((6*self.n_gauss,24))
        self.deri_phi_real=np.zeros((3*self.n_gauss,8))
        self.deri_phi_param=np.zeros((3,self.n_gauss))
        self.Ke=np.zeros((24,24))
        self.gauss_coordinates=np.zeros((8,self.GDL_no_element))
        self.gauss_weight=np.zeros((8,self.GDL_no_element))
        self.get_gauss_parametric_coordinante_and_weight()

#-----------------------------------------------------------------------------    
        
    def inter_element(self):
        """
        Interpolate function 8 nodes isoparametric hexahedron
        """
        self.phi[:,:]=0
        cont=0
        for i in range(self.n_gauss):
            r=self.gauss_coordinates[cont,0]
            s=self.gauss_coordinates[cont,1]
            t=self.gauss_coordinates[cont,2]
                          
            self.phi[cont,0]=(1/8)*(1-r)*(1-s)*(1-t)
            self.phi[cont,1]=(1/8)*(1+r)*(1-s)*(1-t) 
            self.phi[cont,2]=(1/8)*(1+r)*(1+s)*(1-t)
            self.phi[cont,3]=(1/8)*(1-r)*(1+s)*(1-t)
            self.phi[cont,4]=(1/8)*(1-r)*(1-s)*(1+t) 
            self.phi[cont,5]=(1/8)*(1+r)*(1-s)*(1+t)
            self.phi[cont,6]=(1/8)*(1+r)*(1+s)*(1+t) 
            self.phi[cont,7]=(1/8)*(1-r)*(1+s)*(1+t)
            cont+=1
#-----------------------------------------------------------------------------    
    
    def jacobian_element(self):
        """
        Method to calculate the Jacobian, his determinant and the 
        derivative of interpolate functions for all quadrature points
        phi: Interpolation function 
        self.deri_phi_real: Derivative of the interpolation function in relation
                            the real coordinates.
        self.deri_phi_real[0:3,:]= derivatives of phi of the 1° gauss point. 
        self.deri_phi_real[3:6,:]= derivatives of phi of the 2° gauss point. .
        """ 
       
        cd_no1=self.coordinantes_elem[0,:] 
        cd_no2=self.coordinantes_elem[1,:]  
        cd_no3=self.coordinantes_elem[2,:]  
        cd_no4=self.coordinantes_elem[3,:] 
        cd_no5=self.coordinantes_elem[4,:]  
        cd_no6=self.coordinantes_elem[5,:]  
        cd_no7=self.coordinantes_elem[6,:]  
        cd_no8=self.coordinantes_elem[7,:] 
        cont=0
        self.jacobian[:,:]=0
        self.deri_phi_real[:,:]=0
        self.det_Jacobian[:]=0
        for i in range(self.n_gauss):
            r=self.gauss_coordinates[i,0]
            s=self.gauss_coordinates[i,1]
            t=self.gauss_coordinates[i,2]


            dp_rf1_r=(1-s)*(1-t) 
            dp_rf3_r=(1+s)*(1-t) 
            dp_rf5_r=(1-s)*(1+t)
            dp_rf7_r=(1+s)*(1+t)
       
            dp_rf1_s=(1-r)*(1-t)
            dp_rf3_s=(1+r)*(1-t)
            dp_rf5_s=(1-r)*(1+t)
            dp_rf7_s=(1+r)*(1+t)
       
            dp_rf1_t=(1-r)*(1-s)
            dp_rf2_t=(1+r)*(1-s)
            dp_rf3_t=(1+r)*(1+s)
            dp_rf4_t=(1-r)*(1+s)
            
            
            #dx_dr
            self.jacobian[0,0]=(1/8)*((-cd_no1[0]+cd_no2[0])*dp_rf1_r  
                        +(cd_no3[0]-cd_no4[0])*dp_rf3_r
                        +(-cd_no5[0]+cd_no6[0])*dp_rf5_r  
                        +(cd_no7[0]-cd_no8[0])*dp_rf7_r)       
            
            #dx_ds
            self.jacobian[1,0]=(1/8)*((-cd_no1[0]+cd_no4[0])*dp_rf1_s 
                        + (-cd_no2[0]+cd_no3[0])*dp_rf3_s
                       +(-cd_no5[0]+cd_no8[0])*dp_rf5_s  
                       +(-cd_no6[0]+cd_no7[0])*dp_rf7_s)
            #dx_dt
            self.jacobian[2,0]=(1/8)*((-cd_no1[0]+cd_no5[0])*dp_rf1_t 
                                + (-cd_no2[0]+cd_no6[0])*dp_rf2_t
                               + (-cd_no3[0]+cd_no7[0])*dp_rf3_t 
                               + (-cd_no4[0]+cd_no8[0])*dp_rf4_t)
            #dy_dr
            self.jacobian[0,1]=(1/8)*((-cd_no1[1]+cd_no2[1])*dp_rf1_r  
                        + (cd_no3[1]-cd_no4[1])*dp_rf3_r
                       +(-cd_no5[1]+cd_no6[1])*dp_rf5_r 
                       + (cd_no7[1]-cd_no8[1])*dp_rf7_r)
            #dy_ds
            self.jacobian[1,1]=(1/8)*((-cd_no1[1]+cd_no4[1])*dp_rf1_s 
                                 + (-cd_no2[1]+cd_no3[1])*dp_rf3_s
                       +(-cd_no5[1]+cd_no8[1])*dp_rf5_s 
                       + (-cd_no6[1]+cd_no7[1])*dp_rf7_s)
            #dy_dt       
            self.jacobian[2,1]=(1/8)*((-cd_no1[1]+cd_no5[1])*dp_rf1_t
                                 + (-cd_no2[1]+cd_no6[1])*dp_rf2_t
                       + (-cd_no3[1]+cd_no7[1])*dp_rf3_t 
                       + (-cd_no4[1]+cd_no8[1])*dp_rf4_t)
            #dz_dr       
            self.jacobian[0,2]=(1/8)*((-cd_no1[2]+cd_no2[2])*dp_rf1_r 
                                 + (cd_no3[2]-cd_no4[2])*dp_rf3_r
                      +(-cd_no5[2]+cd_no6[2])*dp_rf5_r
                      + (cd_no7[2]-cd_no8[2])*dp_rf7_r)
            #dz_ds
            self.jacobian[1,2]=(1/8)*((-cd_no1[2]+cd_no4[2])*dp_rf1_s 
                                 + (-cd_no2[2]+cd_no3[2])*dp_rf3_s
                       +(-cd_no5[2]+cd_no8[2])*dp_rf5_s 
                       + (-cd_no6[2]+cd_no7[2])*dp_rf7_s)
            #dz_dt      
            self.jacobian[2,2]=(1/8)*((-cd_no1[2]+cd_no5[2])*dp_rf1_t
                                 + (-cd_no2[2]+cd_no6[2])*dp_rf2_t
                       + (-cd_no3[2]+cd_no7[2])*dp_rf3_t 
                       + (-cd_no4[2]+cd_no8[2])*dp_rf4_t)
            
            #Derivative in relation real coordinantes
            self.deri_phi_real[cont:cont+3,:]=self.phi_real_derivative(
                               dp_rf1_r, dp_rf3_r,dp_rf5_r,dp_rf7_r,dp_rf1_s,
                         dp_rf3_s,dp_rf5_s,dp_rf7_s,dp_rf1_t,dp_rf2_t,dp_rf3_t,
                         dp_rf4_t,self.jacobian)       
            #det_Jacobian
            self.det_Jacobian[i]=np.linalg.det(self.jacobian)
            cont+=3
       
#----------------------------------------------------------------------------- 
    def get_gauss_parametric_coordinante_and_weight(self):
        
        """
        Method to get parametric coordinates of the integration points.
        """
        cont=0
        weights=np.array([1,1,1])
        #Unitary weights in this type of integration. Made just to remember. 
        r=np.array([-3**(-1/2), 3**(-1/2)])
        s=np.array([-3**(-1/2), 3**(-1/2)])
        t=np.array([-3**(-1/2), 3**(-1/2)])
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    self.gauss_coordinates[cont,0]=r[i]
                    self.gauss_coordinates[cont,1]=s[j]
                    self.gauss_coordinates[cont,2]=t[k]
                    self.gauss_weight[cont,0]=weights[i]
                    self.gauss_weight[cont,1]=weights[j]
                    self.gauss_weight[cont,2]=weights[k]
                    cont+=1
		
#-----------------------------------------------------------------------------# 
    
    def phi_real_derivative(self,dp_rf1_r, dp_rf3_r,dp_rf5_r,dp_rf7_r,
                                 dp_rf1_s,dp_rf3_s,dp_rf5_s,dp_rf7_s,dp_rf1_t,
                                 dp_rf2_t,dp_rf3_t,dp_rf4_t,parame_derivative):
        """
        Method to assemble the matrix of derivatives of interpolation functions 
        in relation to the parametric coordinates for each Gauss point. 
        """
        self.deri_phi_param[:,:]=0
        
        self.deri_phi_param[0,0]=-dp_rf1_r/8; self.deri_phi_param[0,1]=dp_rf1_r/8
        self.deri_phi_param[0,2]=dp_rf3_r/8; self.deri_phi_param[0,3]=-dp_rf3_r/8
        self.deri_phi_param[0,4]= -dp_rf5_r/8; self.deri_phi_param[0,5]=dp_rf5_r/8
        self.deri_phi_param[0,6]=dp_rf7_r/8; self.deri_phi_param[0,7]= -dp_rf7_r/8
        
        self.deri_phi_param[1,0]=-dp_rf1_s/8; self.deri_phi_param[1,1]=-dp_rf3_s/8
        self.deri_phi_param[1,2]=dp_rf3_s/8; self.deri_phi_param[1,3]=dp_rf1_s/8
        self.deri_phi_param[1,4]=-dp_rf5_s/8; self.deri_phi_param[1,5]=-dp_rf7_s/8
        self.deri_phi_param[1,6]=dp_rf7_s/8; self.deri_phi_param[1,7]=dp_rf5_s/8
        
        self.deri_phi_param[2,0]=-dp_rf1_t/8; self.deri_phi_param[2,1]=-dp_rf2_t/8
        self.deri_phi_param[2,2]=-dp_rf3_t/8; self.deri_phi_param[2,3]=-dp_rf4_t/8
        self.deri_phi_param[2,4]= dp_rf1_t/8; self.deri_phi_param[2,5]=dp_rf2_t/8
        self.deri_phi_param[2,6]=dp_rf3_t/8; self.deri_phi_param[2,7]= dp_rf4_t/8
        
        real_derivative=np.linalg.solve(parame_derivative,self.deri_phi_param)
        
        return real_derivative
        

#-----------------------------------------------------------------------------# 

    def B_element(self):
        """
        Method to compute B matrix for all quadrature points.
        self.B[0:6,:]= B matrix of the first guass point
        self.B[6:12,:]= B matrix of the second guass point .... and so on
        """
        self.B[:,:]=0
        cont1=0
        cont2=0

        for i in range(self.n_gauss):      
            cont3=0
            for j in range(self.n_gauss):
                self.B[cont1,cont3]=self.deri_phi_real[cont2,j]
                self.B[cont1+4,cont3]=self.deri_phi_real[cont2+2,j]
                self.B[cont1+5,cont3]=self.deri_phi_real[cont2+1,j]
            
                self.B[cont1+1,cont3+1]=self.deri_phi_real[cont2+1,j]
                self.B[cont1+3,cont3+1]=self.deri_phi_real[cont2+2,j]
                self.B[cont1+5,cont3+1]=self.deri_phi_real[cont2,j]
            
                self.B[cont1+2,cont3+2]=self.deri_phi_real[cont2+2,j]
                self.B[cont1+3,cont3+2]=self.deri_phi_real[cont2+1,j]
                self.B[cont1+4,cont3+2]=self.deri_phi_real[cont2,j]
                cont3+=3
            cont1+=6
            cont2+=3

#-----------------------------------------------------------------------------# 
            
    def get_Ke_element(self,material,displacement=0): 
        """
        Method to compute elementary stiffness.
        """
        self.Ke[:,:]=0
        cont=0
        for i in range(self.n_gauss):
            #Posso enviar para o modelo constitutivo a informaçao de cada ponto
            #Gauss. Isto é, a matriz B de cada ponto de Gauss e o deslocamento 
            #do elemento recebido de fora. Dentro do modelo constitutivo 
            #Tera uma rotina para atualizar a deformaçao do ponto de Gauss
            #Em seguida vai para o calculo do modulo tangente. 
            #Mesma consideraçao vale para atualizaçao das forças inernas. 
            #A rotina de atualizaçao das forças internas estara dentro da rotina material.
            #tg_modu: Tangent modulus
            #Chamada modelo constitutivo para obter modulo_tangente.
            material.get_tangent_modulus(displacement,self)
            w_i=self.gauss_weight[i,0]
            w_j=self.gauss_weight[i,1]
            w_k=self.gauss_weight[i,2]
            B_t=np.transpose(self.B[cont:cont+6,:])
            self.Ke+=(np.matmul(B_t,(np.matmul(material.tangent_modulus,self.B[cont:cont+6,:])))
                                     *self.det_Jacobian[i]*w_i*w_j*w_k)                      
            cont+=6
                
#-----------------------------------------------------------------------------# 
                   
    def get_coordinantes_nodes_elem(self,mesh):
        """
        Method to allocate element coordinates
        """
        for i in range(self.n_nodes_element):
            n_no=mesh.connectivity[self.element_number,i]
            self.coordinantes_elem[i,:]=mesh.nodes[n_no,:]

