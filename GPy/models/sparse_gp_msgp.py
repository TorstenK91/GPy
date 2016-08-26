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
import warnings

class GPMSGP(Model):
    """
    
    inference_method: grid_gaussian_inference
    kernels: List of kernels which act upon the different grid-dimensions of Z  
    
    TO-DO: interpolation_method:
    TO-DO: Normalizer
    TO-DO: Linearize GP (linearize local interpolation)
     
    
    """
    def __init__(self, X, Y, Z, kernels, name = 'gp_msgp', interpolation_method = None, normalize = False):
        super(GPMSGP,self).__init__(name)
        
        self.X = ObsAr(X) #Not sure what Obsar
        self.Z = Z
        
        if normalize:
            with_mean = True
            with_std = True
        else:
            with_mean = True
            with_std = True
            
        self.normalizer = StandardScaler(with_mean=with_mean,with_std=with_std)
        self.X = self.normalizer.fit_transform(self.X)
        self.Z_normalizers = [None]*len(self.Z)
        self.Z_normalizers = [StandardScaler(with_mean=with_mean,with_std = with_std).fit(X_z) for X_z in self.Z]
        self.Z = [self.Z_normalizers[ii].transform(self.Z[ii]) for ii in range(len(self.Z))]
            
        
        
        self.num_data, self.input_dim = self.X.shape
        
        assert Y.ndim == 2
        #logger.info("initializing Y")
        
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
         
        input_grid_dims = list()
        for X_z in Z:
            input_grid_dims.append(np.shape(X_z)[1])
        self.input_grid_dims = input_grid_dims
         
        self.inference_method = GridGaussianInference()
        
        self.likelihood = likelihoods.Gaussian() #TO-DO: do we even need this?
            
        self.kern = KernGrid(kernels,self.likelihood,input_grid_dims,interpolation_method = interpolation_method)    
        
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
        
        print(self.posterior_prediction["alpha_pred"])
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
        
        
    def predict(self,X_new,pred_var = True):
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
        W_pred = self.kern.getW(X_new) ## get the n* x N Kernel Interpolation matrix
        #print(W_pred)
        #print(alpha_pred)
        mu = W_pred.dot(alpha_pred)
        
        if pred_var:
            nu_pred_var = self.posterior_prediction["nu_pred_var"]
            
            k_self = self.kern.custom_K_diag(X_new)

            sigm =  np.maximum(0, k_self - np.reshape(W_pred.dot(nu_pred_var),(-1,1)))

            return mu, sigm
        
        return np.reshape(mu,(-1,1))
        
    def predictive_gradients(self, X_new,feature_dims = None):
        """
        
        
        
        """

        if len(np.shape(X_new)) == 1:
            X_new = np.reshape(X_new,(1,len(X_new)))
            
        n_data_pred, n_dims = np.shape(X_new)
        
        if feature_dims is None:
            feature_dims = np.arange(n_dims)
            
        alpha_pred = self.posterior_prediction["alpha_pred"]
        
        X_new = self.normalizer.transform(X_new)
        
        Ws_pred_grad = self.kern.getdWdX(X_new,feature_dims = feature_dims)
        
        n_deriv = len(feature_dims)
        pred_grad = np.zeros((n_data_pred,n_deriv))
        for ii in range(n_deriv):
            pred_grad[:,ii] = Ws_pred_grad[ii].dot(alpha_pred)
            
        return pred_grad
        
        
    
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

    