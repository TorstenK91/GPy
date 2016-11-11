# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 18:21:24 2016

@author: student
"""

import numpy as np
from GPy.kern import KernGrid,Kern
from GPy.core.model import Model
from paramz import ObsAr
from GPy import likelihoods
from GPy.inference.latent_function_inference.grid_gaussian_inference import GridGaussianInference
from GPy.mappings import Constant
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from GPy.util import Timer

import warnings

class GPMSGP(Model):
    """
    
    inference_method: grid_gaussian_inference
    kernels: List of kernels which act upon the different grid-dimensions of Z  
    
    TO-DO: interpolation_method:
    TO-DO: Normalizer
    TO-DO: Linearize GP (linearize local interpolation)
     
    
    """
    def __init__(self, X, Y, Z, kernels, name = 'gp_msgp', interpolation_method = None,grid_dims = None, normalize = False):      
        super(GPMSGP,self).__init__(name)
        
        self.X = ObsAr(X) #Not sure what Obsar
        
        
            
        if grid_dims is None:
                dims = [None]*len(Z)
                max_dim_ii = 0
                for ii in range(len(Z)):
                    dims[ii] = np.arange(max_dim_ii,max_dim_ii+np.shape(Z[ii])[1])
                    max_dim_ii = dims[ii-1][-1]+1
                grid_dims = dims
        else:
            grid_dims_to_create_id = []
            grid_dims_create = []
            grid_create_args = []
            n_grid_dims = len(grid_dims)
            for ii in range(n_grid_dims):
                if isinstance(Z[ii],dict):
                    grid_dims_to_create_id.append(ii)
                    grid_dims_create.append( grid_dims[ii])
                    grid_create_args.append(Z[ii])
            
            if len(grid_dims_to_create_id) > 1:
                Z_create = self.create_grid(grid_create_args,grid_dims = grid_dims_create)
                
                for ii in range(len(grid_dims_to_create_id)):
                    Z[grid_dims_to_create_id[ii]] = Z_create[ii]
        
        self.Z = Z
        self.input_grid_dims = grid_dims
                
        """
        if isinstance(Z,dict): #automatically create the grid
            Z,self.input_grid_dims = self.create_grid(Z,grid_dims = grid_dims)
            self.Z = Z
        else:
            
            self.input_grid_dims = grid_dims
            self.Z = Z
        """

        if normalize:
            with_mean = True
            with_std = True
        else:
            with_mean = False
            with_std = False
            
        self.normalizer = StandardScaler(with_mean=with_mean,with_std=with_std)
        self.X = self.normalizer.fit_transform(self.X)
        self.Z_normalizers = [None]*len(self.Z)
        
        self.Z_normalizers = [StandardScaler(with_mean=with_mean,with_std = with_std).fit(X_z) for X_z in self.Z]
        self.Z = [self.Z_normalizers[ii].transform(self.Z[ii]) for ii in range(len(self.Z))]
            
        
        
        self.num_data, self.input_dim = self.X.shape
        
        assert Y.ndim == 2

        self.Y = ObsAr(Y)
        
        self.Y_metadata = None #TO-DO: do we even need this?
        
        assert np.shape(Y)[0] == self.num_data
        
        _,self.output_dim = self.Y.shape
        
        
        
        #check if kernels is a list or just a single kernel
        #and then check if every object in list is a kernel
        
        try:
            for kernel in kernels:
                assert isinstance(kernel,Kern)
            
        except TypeError:
            assert isinstance(kernels,Kern)
            kernels = list([kernels])
         
        
         
        self.inference_method = GridGaussianInference()
        
        self.likelihood = likelihoods.Gaussian() #TO-DO: do we even need this?
            
        self.kern = KernGrid(kernels,self.likelihood,self.input_grid_dims,interpolation_method = interpolation_method)    
        
        self.mean_function = Constant(self.input_dim,self.output_dim)
        self.kern.update_Z(Z)
        ##for test set n_neighbors = 4
        self.kern.init_interpolation_method(n_neighbors = 8)
        self.kern.update_X_Y(X,Y)
        
        ## register the parameters for optimization (paramz)
        self.link_parameter(self.kern)
        self.link_parameter(self.likelihood)
        
        
        
        ## need to do this in the case that someone wants to do prediction without/before 
        ## hyperparameter optimization
        self.parameters_changed()
        self.posterior_prediction = self.inference_method.update_prediction_vectors(self.kern,self.posterior,self.grad_dict,self.likelihood)

    def log_likelihood(self):
        return np.real(self._log_marginal_likelihood)
        
    def parameters_changed(self):
        """
        
        """
        
        self.kern.parameters_changed()
        # update marginal likelihood
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, 
                                                                                                        self.likelihood,
                                                                                                        mean_function = self.mean_function)
        
        # update kernel hyperparameter gradients AND gaussian variance hyperparameter      
        ##TO-DO: Why does Vs,Es not work as positional args. Am I stupid?
             
        self.kern.custom_update_gradients_full(self.posterior["alpha"], self.grad_dict["V"],self.grad_dict["E"],self.grad_dict["S"]) 
        
        
    def predict(self,X_new,pred_var = True, W_pred = None):
        """
        Test data prediction:
        
        alpha_pred is precomputed as:
        alpha_pred = K_Z*W*alpha, where alpha is the alpha similar
        to the standard gp
        
        mu is then computed as:
        m = W_pred*alpha_pred
        where W_pred is supposed to be extremely sparse (<10 nonzeros)
        
        efficient sparse matrix vector product then leads to O(1) prediction
        

        """

        alpha_pred = self.posterior_prediction["alpha_pred"]
        
        X_new = self.normalizer.transform(X_new)
        if W_pred is None:
            W_pred = self.kern.getW(X_new) ## get the n* x N Kernel Interpolation matrix
            
        
        #print(W_pred)
        #print(alpha_pred)
        with Timer() as t:
            mu = W_pred.dot(alpha_pred)
        print("Runtime mu: {} s".format(t.secs))
        
        if pred_var:
            nu_pred_var = self.posterior_prediction["nu_pred_var"]
            
            k_self = self.kern.custom_K_diag(X_new)

            sigm =  np.maximum(0, k_self - np.reshape(W_pred.dot(nu_pred_var),(-1,1)))

            return mu, sigm
        
        return np.reshape(mu,(-1,1))
        
    def predictive_gradients(self, x_new,feature_dims = None,predict_mean = True, predictive_var = False):
        """
        Compute the gradients of the predictive mean of a SINGLE test input w.r.t. the input dimensions.
        Per default the predictive mean itself is computed too.
        
        input: 
            x_new
        
        
        
        """

        if len(np.shape(x_new)) == 1:
            x_new = np.reshape(x_new,(1,len(x_new)))
            
        n_data_pred, n_dims = np.shape(x_new)
        
        if feature_dims is None:
            feature_dims = np.arange(n_dims)
            
        alpha_pred = self.posterior_prediction["alpha_pred"]
        
        x_new = self.normalizer.transform(x_new)
        
        if predict_mean:
            Ws_pred_grad, W_pred = self.kern.getdWdX(x_new,feature_dims = feature_dims,compute_W = True)
            
            print(W_pred)
            if predictive_var:
                pred_mu,pred_var = self.predict(x_new,pred_var=True,W_pred=W_pred)
            else:
                pred_mu = self.predict(x_new,pred_var = False,W_pred = W_pred)
            print(pred_mu)
        else:
            
            Ws_pred_grad = self.kern.getdWdX(x_new,feature_dims = feature_dims)
        
        n_deriv = len(feature_dims)
        pred_grad = np.zeros((n_data_pred,n_deriv))
        for ii in range(n_deriv):
            pred_grad[:,ii] = Ws_pred_grad[ii].dot(alpha_pred)
            
            
        if predict_mean:
            if predictive_var:
                return pred_grad,pred_mu,pred_var
            else:
                return pred_grad,pred_mu
                
        return pred_grad
        
        
    def forward_trajectory(self,x0,U,predictive_var = False):
        """
        Compute a forward trajectory with the msgp model
        
        Input: 
            x0: ns x 1 vector for the initial state
            U:  nu x T matrix of control inputs for T steps
            
        Output:
            Mu: ns x T matrix of the predictiv means
            Var: ns x T matrix of the predictive variances
            dS : list of T matrices of shape (ns x ns) containing the linearization w.r.t the state dimensions
            dU : list of T matrices of shape (ns x nu) containing the linearization w.r.t the control dimensions
             
        """
        ns = np.shape(x0)[0]
        nu, T = np.shape(U) 
        
        x_ii = x0.T
        
        
        
    
        grad_state = np.zeros((T,ns,ns))
        grad_control = np.zeros((T,ns,nu))
        states = np.zeros((T,ns))
        variances = np.zeros((T,ns))


        for ii in range(T):
            
            #state_action = x_ii
            state_action = np.append(x_ii,U[:,ii]).reshape((1,-1))
            print(np.shape(state_action))

            
            if predictive_var:
                new_state_grads,new_state,new_state_var = self.predictive_gradients(state_action,predictive_var = True)
                variances[ii] = new_state_var
            else:
                
                new_state_grads,new_state = self.predictive_gradients(state_action,predictive_var = False)
            print(new_state_grads)
            print(new_state)
            grad_state[ii,:,:] = new_state_grads[:,np.arange(ns)]
            grad_control[ii,:,:] = new_state_grads[:,np.arange(ns+1,ns+nu)]

            states[ii,:] = new_state

            x_ii = new_state.T
            
        if predictive_var:
            return grad_state,grad_control,states,variances
        
        return grad_state,grad_control,states
            

                  
        
        
    def create_grid(self,arg_list,grid_dims = None):
        """
        Create the multidimensional grid based on
        the specifications provided in arg_dict
        """
        method = arg_list[0]["method"]
        if method == "kmeans":
            return self._create_kmeans_grid(grid_dims,arg_list[0]["n_data_per_grid"])
        elif method == "dummy":
            pass
        else:
            raise ValueError("Wrong grid creation method chosen! "
            +"Make sure that either Z already contains the grid data "
            +"or its a dict containing necessary information for the grid data creation")
        
    def _create_kmeans_grid(self,grid_dims,n_data_per_grid):
        """
        
        """
        
        grid_cluster = KMeans(n_clusters = n_data_per_grid)
        grid_cluster.fit(self.X)
        
        grid_cluster_centers = grid_cluster.cluster_centers_
        
        n_grid_dims = len(grid_dims)
        Z = []
        
        for ii in range(n_grid_dims):
            Z.append(grid_cluster_centers[:,grid_dims[ii]])
            
        return Z
            
    def optimize(self, optimizer=None, start=None, messages=False, max_iters=1000, ipython_notebook=True, clear_after_finish=False, **kwargs):
        """
        Optimize the model using self.log_likelihood and self.log_likelihood_gradient, as well as self.priors.
        kwargs are passed to the optimizer. They can be:

        :param max_iters: maximum number of function evaluations
        :type max_iters: int
        :messages: whether to display during optimisation
        :type messages: bool
        :param optimizer: which optimizer to use (defaults to self.preferred optimizer), a range of optimisers can be found in :module:`~GPy.inference.optimization`, they include 'scg', 'lbfgs', 'tnc'.
        :type optimizer: string
        :param bool ipython_notebook: whether to use ipython notebook widgets or not.
        :param bool clear_after_finish: if in ipython notebook, we can clear the widgets after optimization.
        """
        self.inference_method.on_optimization_start()
        try:
            super(GPMSGP, self).optimize(optimizer, start, messages, max_iters, ipython_notebook, clear_after_finish, **kwargs)
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught, calling on_optimization_end() to round things up")
            self.inference_method.on_optimization_end()
            raise
            
        self.posterior_prediction = self.inference_method.update_prediction_vectors(self.kern,self.posterior,self.grad_dict,self.likelihood)

    