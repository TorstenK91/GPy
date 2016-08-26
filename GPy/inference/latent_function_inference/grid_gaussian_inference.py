# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:55:26 2016

@author: student
"""
import numpy as np
from ...util import msgp_linalg
from scipy import linalg
from . import LatentFunctionInference
from GPy import likelihoods
from ...util.timer import Timer
class GridGaussianInference(LatentFunctionInference):
    """
    An object for inference when likelihood is Gaussian and data is located
    on a grid.
    
    The class uses kronecker and toeplitz structure for highly scalable 
    inference.
    
    Details can be found in:
    
    
    
    
    
    """
    def __init__(self):
        """
        Not sure about this yet
        
        """
        
    
    def inference(self, kern, likelihood, mean_function = None,W=None, Y_metadata=None,precision = None):
        
        """
        Not sure about kronmagic yet
        """
        
        
        """
        Not sure about design of kern yet. 
        Needs to contain something like "is_toeplitz" per grid dimension
        and has to provide p different Kernel-matrices (p = #grid dimensions)
        
        
        
        
        ????? Do we even need partial grids for MSGP????? Probably not
        TO-DO: Incomplete grids:
                    The M-matrix is usually used to lift incomplete grid information
                    to a complete grid (y-values only for parts of the grid)
        """
        if not isinstance(likelihood,likelihoods.Gaussian):
            raise NotImplementedError("Not sure what todo if likelihood is non-gaussian")
        
        
        if kern.name != 'kern_grid':
            raise ValueError("kernel type has to be kern_grid")
            
        
        Z = kern.Z
        X = kern.X
        Y = kern.Y 
        W_x_z = kern.W_x_z

        kronmagic = False 
        
        if mean_function is None: 
            m = 0
        else:
            m = mean_function.f(X)
        
        if precision is None:
            precision = likelihood.gaussian_variance()
        

        p = len(Z) 
        n_grid = 1 
        n_data = np.shape(Y)[0]
        D=0
        for i in range(p):
            n_grid=n_grid*np.shape(Z[i])[0]
            D = D+np.shape(Z[i])[1]
        
        K  = kern.get_K_Z()

        is_toeplitz = kern.get_toeplitz_dims()
        #xe = np.dot(M,xe)
        #n=np.shape(xe)[1]
        for j in range(len(K)):
            if is_toeplitz[i] == True:
                K[j] = linalg.toeplitz(K[j])
                
        sn2 = np.exp(2*precision) 
        
        V = []
        E = []
    #with Timer() as t:
        for j in range(len(K)):
            if is_toeplitz[i] == True:
                V_j,E_j = msgp_linalg.eigr(linalg.toeplitz(K[j]))
            else:
                V_j,E_j = msgp_linalg.eigr(K[j])

            V.append(np.matrix(V_j))
            E.append(np.matrix(E_j))#
    #print("Runtime eigendecomposition: {}".format(t.secs))
        
        e = np.ones((1,1))
        for ii in range(len(E)):
              e = linalg.kron(np.matrix(np.diag(E[ii])).T,e) 
        e = np.ravel(e.astype(float))
        if kronmagic:
            """
            Problem: here we would need the eigendecomposition of the kernel
            matrices. We would need to get those from the kernels.
            
            """
            raise NotImplementedError
            #s = 1/(e+sn2)
            #order = range(1,N)
        else:

            sort_index = np.argsort(e)

            sort_index = sort_index[::-1]
            sort_index = np.ravel(sort_index)
            
            eord = e[sort_index]

            """
            if n<N: ##We dont need this here <- only for partial grid structure
                eord = np.vstack((eord,np.zeros((n-N),1)))
                order = np.vstack((order,range(N+1,n).T))
            """

            s = 1./((float(n_data)/float(n_grid))*eord[0:n_data]+sn2) 

           
           
        if kronmagic:
        ## infGrid.m line 61
            raise NotImplementedError
        else:
            
            kron_mvm_func = lambda x: msgp_linalg.mvm_K(x,K,sn2,W_x_z)
            shape_mvm = (n_data,n_data) ## To-Do: what is the shape of the mvm?
            L = lambda x: -msgp_linalg.solveMVM(x,kron_mvm_func,shape_mvm,cgtol = 1e-5,cgmit=1000) # ein - zuviel?
        
    #with Timer() as t:
        alpha = -L(Y-m) 
    #print("Computation of alpha in likelihood comp: {}".format(t.secs))
        
        #print(np.shape(W_x_z.dot(WTFWTFWTF.T).T))
     
        #print(np.shape(Y-m))
        #print(alpha_test)
      
        #alpha = alpha_test
        lda = -np.sum(np.log(s))
        print(lda)
        #lda = -2*sum(np.log(np.diag(linalg.cholesky(W_x_z.dot(W_x_z.dot(msgp_linalg.kronmvm(K,np.eye(n_grid))).T).T+sn2*np.eye(n_data)))));

        #log_marginal = np.dot((Y-m).T,alpha)/2 + n_data*np.log(2*np.pi)/2 + lda/2 
        print("model fit: {}".format(-np.dot((Y-m).T,alpha)/2))
        print("complexity: {}".format(lda/2))
        log_marginal = -np.dot((Y-m).T,alpha)/2 - n_data*np.log(2*np.pi)/2 - lda/2
        #print(log_marginal)
            
        #print(alpha)
        #print(W_x_z.T)
        #print(np.max(K[0]))
        #alpha_pred = msgp_linalg.kronmvm(K,W_x_z.T.dot(alpha)) # we need the term K_Z_Z*W.T*alpha for the predictive mean
        
        """
        Calculate  the nu-term for predictive variance
        
        TO-DO: not sure if the K_tilde kronmvm is correct
        """    

        
    
        grad_dict = dict()
        grad_dict["V"] = V 
        grad_dict["E"] = E
        grad_dict["S"] = dict()
        grad_dict["S"]["s"] = s
        grad_dict["S"]["eord"] = eord
        grad_dict["S"]["sort_index"] = sort_index
        post = dict()
        post["alpha"] = alpha
        
        
        return post, log_marginal, grad_dict
        
        
    def update_prediction_vectors(self,kern,posterior,grad_dict,likelihood):
        """
        
        """
        
        
        V= grad_dict["V"]
        E = grad_dict["E"]
        
        alpha = posterior["alpha"]
        K = kern.get_K_Z()
        
        sn2 = np.exp(2*likelihood.gaussian_variance() )        
        
        W_x_z = kern.W_x_z
        
        n_data , n_grid = np.shape(W_x_z)
        
        alpha_pred = msgp_linalg.kronmvm(K,W_x_z.T.dot(alpha))
        
        K_tilde = [None]*len(K)
        for jj in range(len(K)):
            K_tilde[jj] = np.dot(np.dot(V[jj],np.sqrt(E[jj])),V[jj].T)

        
        n_s = 20
        res_s = np.zeros((n_data,n_s))        
        for kk in range(n_s):
            ## create normal distributed samples in batches since they are iid anyway
            n_batch = 100;
            m_it = n_grid // n_batch    
            m_rem = n_grid % n_batch
            g_m = np.zeros((n_grid,))
            for ii in range(m_it):
                g_m_ii = np.random.multivariate_normal(np.ones((n_batch,)),np.eye(n_batch))
                g_m[n_batch*ii:n_batch*(ii+1)] = g_m_ii
            if m_rem > 0:
                g_m[n_batch*m_it:] = np.random.multivariate_normal(np.ones((m_rem,)),np.eye(m_rem))
            
            n_it = n_data // n_batch
            n_rem = n_data % n_batch
            
            g_n = np.zeros((n_data,))
            for ii in range(n_it):
                g_n_ii = np.random.multivariate_normal(np.ones((n_batch,)),np.eye(n_batch))
                g_n[n_batch*ii:n_batch*(ii+1)] = g_n_ii
            
            if n_rem > 0:
                g_n[n_batch*n_it:] = np.random.multivariate_normal(np.ones((n_rem,)),np.eye(n_rem))
           
            g_n = np.reshape(g_n,(-1,1))
            g_m = np.reshape(g_m,(-1,1))
    
            z = W_x_z.dot(msgp_linalg.kronmvm(K_tilde,g_m)) + sn2*g_n
    
            kron_mvm_func = lambda x: msgp_linalg.mvm_K(x,K,sn2,W_x_z)
            shape_mvm = (n_data,n_data)
            res_s[:,kk] = msgp_linalg.solveMVM(z,kron_mvm_func,shape_mvm,cgtol = 1e-5,cgmit=1000)
        
        ### see equation (10) in "thoughts on msgp"
        
        nu_pred_var = np.mean(msgp_linalg.kronmvm(K,W_x_z.T.dot(res_s))**2,axis = 1)
        
        posterior_prediction = dict()
        posterior_prediction["alpha_pred"] = alpha_pred
        posterior_prediction["nu_pred_var"] = nu_pred_var
        
        return posterior_prediction