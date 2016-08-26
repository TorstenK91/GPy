
import numpy as np

from .kern import CombinationKernel
from ...util import msgp_linalg
from ...core.parameterization import Param
from numpy import matlib
from ...util.grid_interpolation import NNRegression
from scipy import linalg
from ...util.timer import Timer


import warnings

class KernGrid(CombinationKernel):
    """
    Meta-class for kernel of data on a grid structure, where
    each grid dimension has its own kernel
    
    TO-DO: Not sure yet how to combine this with the slicing operation
    
    TO-DO: As of now kernels can only be applied to a whole grid-dimension.
           This is not necessary. If a grid-dimension consists of multiple
           data dimensions (i.e. grid dimension i: input_grid_dims[i] > 1), 
           we can apply multiple kernels on subsets of the data dimensions of a 
           grid-dimension. !!!
    
    """
    
    def __init__(self,kernels,likelihood,input_grid_dims,useGPU = False, interpolation_method = None):
        super(KernGrid,self).__init__(kernels,'kern_grid')
        
        ## Data dimensions per grid dimension
        self.input_grid_dims = input_grid_dims
        
        ## Number of grid dimensions
        self.q = len(input_grid_dims)
        
        ## The kernels. Each kernel works on one grid dimension
        self.kernels = kernels
        
        ## the variance parameter sn2
        """
        TO-DO: not sure about transformation Logexp?
        self.variance = Param('variance', variance, Logexp()) 
        """
        self.likelihood = likelihood        
        
        
        if not len(kernels) == self.q:
            raise ValueError("We need the same number of kernels as we have grid dimensions")
        
                #Create the n by N interpolation matrx (n: size of trainingset X, N: number of gridpoints)
        
        
        self.X = None 
        self.Y = None
        self.Z = None
        self.W_x_z = None
        self.Ks = None
        self.Z_all = None
        
        #self._calc_Z_all()
        self.interpolator = None
        if interpolation_method is None:
            self.interpolation_method = "NNRegression"
        else:
            self.interpolation_method = "NNRegression"
            
        self.K_is_up_to_date = False
        #self.W_is_up_to_date = False
        
    def update_X_Y(self,X,Y):
        """
        Update the training data, which requires recomputation of the interpolation matrtices
        (The interaction between X and Z, where )
        """
        self.X = X 
        self.Y = Y 
        self.n_data = np.shape(Y)[0]
        
      
        self.W_x_z = self.interpolator.getWeights(X)

            
    def init_interpolation_method(self,*args,**kwargs):
    
        if self.Z_all is None:
            raise ValueError("Need to initialize Z before initializing interpolator")
        if self.interpolation_method == "NNRegression":
            self.interpolator = NNRegression(Z=self.Z_all,*args,**kwargs)
        #elif ...:
        else:
            raise NotImplementedError
    def _calc_Z_all(self):
        """
        Calc the full grid data from the submatrices via cross products
        
        """
        
        Z_all = self.Z[0]
        for ii in range(1,self.q):
            Z_all = self._cartesian_simple(Z_all,self.Z[ii])
        
        self.Z_all = Z_all
        
    def update_Z(self,Z):
        """
        Update the grid-data which requires the recomputation of 
        
    
        """
        self.Z = Z
        self.K_is_up_to_date = False
        #if not lazy_recomputation:
        self._K_Z()
        self._calc_Z_all()
        
        p=len(Z)
        n_grid = 1
        D=0
        for i in range(p):
            n_grid=n_grid*np.shape(Z[i])[0]
            D = D+np.shape(Z[i])[1]
        self.n_grid = n_grid
        
        if not self.interpolator is None: ##this can happen when initing Z the first time
            self.interpolator.update_Z(Z)
    
    def custom_K(self,X,X2 = None):
        """
        Have to call this custom_K since K uses the slicing operations
        TO-DO: this is terrible
        """
        
        if X2 is None:
            X2 = X
              
        K_all = np.zeros((np.shape(X)[0],np.shape(X2)[0]))
        
        dims = [None]*self.q
        
        dims[0] = np.arange(self.input_grid_dims[0])
        for ii in np.arange(1,self.q):
            
            max_dim_ii = dims[ii-1][-1]+1
            dims[ii] = np.arange(max_dim_ii,max_dim_ii+self.input_grid_dims[ii])
        
        for ii in range(self.q):
            K_all =  np.multiply(K_all , self.kernels[ii].K(X[:,dims[ii]],X2[:,dims[ii]]))
            
        return K_all
    
    def custom_K_diag(self,X):
        """
        Have to call this custom_K since K uses the slicing operations
        TO-DO: this is terrible
        """
              
        K_all = np.ones((np.shape(X)[0],1))
        
        dims = [None]*self.q
        
        dims[0] = np.arange(self.input_grid_dims[0])
        for ii in np.arange(1,self.q):
            
            max_dim_ii = dims[ii-1][-1]+1
            dims[ii] = np.arange(max_dim_ii,max_dim_ii+self.input_grid_dims[ii])
        
        for ii in range(self.q):
            K_ii = np.reshape(self.kernels[ii].Kdiag(X[:,dims[ii]]),(-1,1))
            K_all =  np.multiply(K_all , K_ii)
            
        return K_all
        
    def _K_Z(self):
        """
        
        
        """
        
        self.Ks = self.K_grid(self.Z)
        self.K_is_up_to_date = True
        return self.Ks
    def get_K_Z(self):
        """
        
        
        """
        if self.K_is_up_to_date:
            return self.Ks
        else:
            return self._K_Z()
        
    def K_grid(self, Z,Z2=None):
        """
        Kernel function applied on grid-data Z
        
        input:
            X: list of length q, where q is the number of grid dimension.
               Each X[i] is a data matrix of size n_i x input_grid_dims[i]
        returns:
            Ks: list of q kernel matrices, where Ks[i] is of size n_i x n_i
        """
        
        if not len(Z) == self.q:
            raise ValueError("Data has to have as many data-matrices as grid dimensions")
            
        #collect number of datapoints on grid
        N = 1
        for ii in range(self.q):
            N = N*np.shape(Z[ii])[0]
        
        # Calculate the covariance matrices of each grid-dimension
        Ks = [] 

        for ii in range(self.q):
            Ks.append(np.matrix(self.kernels[ii].K(Z[ii]),dtype='float64'))

        return Ks
        
    def parameters_changed(self):
        """
        This needs to be called if the hyperparameters of the covariance
        functions change (recomputation of Ks)
        
        """
        self.K_is_up_to_date = False
        
    def expandgrid(X):
        """
        Expand grid-data X, which is organized as a list of data-matrices (one data-matrix
        per grid-dimension) to become a full N x N data-matrix where N = n_1 * ... * n_q
         
        TO-DO not sure right now if this is necessary
        """
        raise NotImplementedError
    
    
    def getW(self,X):
        """
        Get the weight matrix for interpolation between grid Z and 
        the input X 
        """
        
        return self.interpolator.getWeights(X) #shit, we probably need 
                                                              #the whole grid 
        
    def getdWdX(self,X,feature_dims = None):
        """
        Get the derivative of the weight matrix for interpolation between grid Z and 
        the input X w.r.t X
        
        """
        return self.interpolator.dW_dX(X,feature_dims = feature_dims)
    def get_toeplitz_dims(self):
        """
        Get boolean vector of length p where 
        is_toepliz[ii] == True when the grid dimension ii
        has toeplitz structue (if it is a regular grid)
        """
        
        warnings.warn("toeplitz dimension selection not implemented!!")
        return np.zeros((self.q,1),dtype=bool)
        
    def custom_update_gradients_full(self,alpha,Vs, Es,S):
        """
        TO-DO: Problem: the current framework does not really fit into
        kernel update scheme, since it uses the product rule 
        to calculate dLdK and then dKdtheta. Thats not efficient in this case        
        
        
        Update the derivatives of negative(!) log likelihood w.r.t. the subkernel hyperparameters.
        Since we have grid-structe, we can use kronecker matrix algebra
        to calculate the derivatives. For mor Details see:
        
        "Scalable Inference for Structured Gaussian Processe Models" -Yunus Satci
        p 134 ff.      
        specifically: we follow Algorithm 16 in the thesis
        
        update gradients w.r.t. all hyperparamters of the kernel matrices        
        
        TO-DO: Propagate gradients to the single grid-dimensions 
                --> need to get dK_dK_i somewhere
        
        
        
        TO-DO: Do we want to compute the Ks,Vs,Es here? since they are required
        to calculate the new log_marginal anyway in GP.parameters_changed
        the log_marginal is updated before the gradients of the kernel is updated
        """
        
        ## required ingredients:
        # gammas: gammas[ii] =  diag(Vs[ii]^T*dKi_dtheta_i*Vs[ii]) 
        # dK_dtheta_i: as in eq 5.36
        alpha = np.reshape(alpha,(-1,1))
        
        precision = self.likelihood.gaussian_variance()
        sn2 = np.exp(2*precision)
        
        #collect all Vs, DK_dtheta and gammas
        dL_dthetas = [None]*self.q # dict such that: dK_dthetas[ii]["<name>"] = dKii_d<name>
                               # the ii-th cov-matrix derivative w.r.t. to the hyperparameter
                               # <name>
        gammas = [None]*self.q
        
        
        Ks = self.get_K_Z()
        
        """
        if Vs is None or Es is None:
            Vs = Es = [None]*self.q
            for ii in range(self.p): 
            
                [Vs[ii],Es[ii]] = msgp_linalg.eigr(Ks[ii]) #DO we need the Es?
            
        """   
            
        for ii in range(self.q) :
        # turns out that formula below is the same as:
        # gammas[ii]= diag(np.dot(Vs[ii].T,np.dot(Ks[ii],Vs[ii])))
        # but generally faster
            """
            TO-DO: Fast formula NOT working as of now
            """
            
            gammas[ii] = np.ravel(np.sum(np.multiply(np.dot(Vs[ii].T,Ks[ii]),Vs[ii].T),axis = 1))
            #gammas[ii] = np.diag(np.dot(Vs[ii].T,np.dot(Ks[ii],Vs[ii])))
            
            
            
        n_data = len(self.X[:,0])
        s = S["s"] #The inverse eigenvalues with sn2 
        eord = S["eord"]
        sort_index = S["sort_index"]
        #diag((Eig_all+sn2*I_N)^(-1)) first part of eq 5.35
        for ii in range(self.q):
            
        #with Timer() as t:
            dKii_dthetas = self.kernels[ii].dK_dTheta(self.Z[ii]) 
        #print("Runtime of local gradients: {} ".format(t.secs))
            for hyp,dKii_dhyp in dKii_dthetas.items():
                
                hyperparam = getattr(self.kernels[ii], hyp)
                ##check if hyp is a list or just one-dimensional -> we might be able to make this faster in future
                if hasattr(hyperparam.gradient,'__len__') and len(hyperparam.gradient) > 1:

                    for kk in range(len(dKii_dhyp)): #iterate over single elements of hyperparam vector
                        gamma_dKii_dhyp = np.ravel(np.sum(np.multiply(np.dot(Vs[ii].T,dKii_dhyp[kk]),Vs[ii].T),axis=1))
                        #gamma_dKii_dhyp = np.diag(np.dot(Vs[ii].T,np.dot(dKii_dhyp[kk],Vs[ii])))

                        gamma_hyp = 1
                    #with Timer() as t:
                        for jj in range(self.q): 
                            if jj == ii:  
                                gamma_hyp = np.kron(gamma_hyp, gamma_dKii_dhyp)
                            else:
                                gamma_hyp = np.kron(gamma_hyp, gammas[jj])
                    #print("Runtime of gamma hyp: {} ".format(t.secs))
                        tmp_list = Ks[0:ii]
                        
                        tmp_list.append(dKii_dhyp[kk])
        
                        kron_matrices_ii = tmp_list+Ks[ii+1:self.q]  # concat list of matrices
                        
                        #line 15 in algo 16
        
                        
                        
                        #need to include W_x_z
                        alpha_long = self.W_x_z.T.dot(alpha)
                    #with Timer() as t:
                        kappa_ii = msgp_linalg.kronmvm(kron_matrices_ii,alpha_long,x_sparse=False)
                    #print("Runtime of kronmvm in gradients: {} ".format(t.secs))
                    
                        """
                        
        
                        TO-DO: not sure if alpha is already in the grid space
                        bring together everything:
                        """
                        #maybe store this?
                        
                        
                        kappa_ii = np.reshape(kappa_ii,(-1,1))
                        
                        #print(sort_index)
                        #print(gamma_hyp[sort_index[0:n_data]])

                        gamma_hyp_dataspace = (float(n_data)/float(self.n_grid))*gamma_hyp[sort_index[0:n_data]]
                        
                        
                
                        ##this is the update step
                        
                        #hyperparam.gradient = dL_dthetas[ii][hyp]
                    #with Timer() as t:
                        hyperparam.gradient[kk] = -0.5*(-np.sum(alpha_long*kappa_ii) + np.sum(gamma_hyp_dataspace*s))
                    #print("Runtime of last computation in gradients: {} ".format(t.secs))
                else:
                    # turns out that formula below is the same as:
                    # gammas[ii][hyp] = diag(np.dot(Vs[ii].T,np.dot(dKii_dhyp,Vs[ii])))
                    # but generally faster
                    
                    """
                    TO-DO: Fast formula NOT working as of now
                    why Dafuq is this different :O :O
                    """
                    gamma_dKii_dhyp = np.sum(np.multiply(np.dot(Vs[ii].T,dKii_dhyp),Vs[ii].T),axis=1)
                    gamma_dKii_dhyp = np.ravel(gamma_dKii_dhyp)
                    #print("Wrong gamma_dKii_dhyp for hyperparameter {}".format(hyp))
                    #print(gamma_dKii_dhyp_test[-5:-1])
                    #gamma_dKii_dhyp = np.diag(np.dot(Vs[ii].T,np.dot(dKii_dhyp,Vs[ii])))
                    #print(np.shape(gamma_dKii_dhyp))
                    #print("gamma_dKii_dhyp for hyperparameter {}".format(hyp))
                    #print(gamma_dKii_dhyp[-5:-1])
                    #print(gamma_dKii_dhyp)
                    #print("testest")
                    """
                    TO-DO: problem: need to include the w_x_z in gamma_hyp    
                    """
                    
                    gamma_hyp = 1
                    
                    for jj in range(self.q): 
                        if jj == ii:  
                            gamma_hyp = np.kron(gamma_hyp, gamma_dKii_dhyp)
                        else:
                            gamma_hyp = np.kron(gamma_hyp, gammas[jj])
                    #print(gamma_hyp)
                    # can Ks[ii+1:self.p-1] be a problem? -> think not, because Ks[p:p-1] = []
    
                    tmp_list = Ks[0:ii]
                    
                    tmp_list.append(dKii_dhyp)
    
                    kron_matrices_ii = tmp_list+Ks[ii+1:self.q]  # concat list of matrices
                    
                    #line 15 in algo 16
    
                    
                    
                    #need to include W_x_z
                    alpha_long = self.W_x_z.T.dot(alpha)
                    kappa_ii = msgp_linalg.kronmvm(kron_matrices_ii,alpha_long,x_sparse=False)
                    
                    
                    """
                    
    
                    TO-DO: not sure if alpha is already in the grid space
                    bring together everything:
                    """
                    #maybe store this?
    
                    
                    kappa_ii = np.reshape(kappa_ii,(-1,1))
                    
                    #print(sort_index)
                    #print(gamma_hyp[sort_index[0:n_data]])
                    gamma_hyp_dataspace = (float(n_data)/float(self.n_grid))*gamma_hyp[sort_index[0:n_data]]
                    
                    ##this is the update step
                    
                    
                    hyperparam.gradient = -0.5*(-np.dot(alpha_long.T,kappa_ii) + np.dot(gamma_hyp_dataspace.T,s))
                
        #precision.gradient = sn2 * (sum(s) - np.dot(alpha.T,alpha)) #no need to lift alpha here       
        precision.gradient = -sn2 * (sum(s) - np.dot(alpha.T,alpha))
        
    
    def _cartesian_simple(self,x,y):
        """
        Compute cartesian product between two grid-dimensions
        """
        
        n1,d1 = np.shape(x)
        n2,d2 = np.shape(y)
        cart_prod = np.empty((0,d1+d2), float)
        
        for ii in range(n1):
            cart_prod = np.vstack((cart_prod,np.hstack((matlib.repmat(x[ii],n2,1),y))))
        return cart_prod    

