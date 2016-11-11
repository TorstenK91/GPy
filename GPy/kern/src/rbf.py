# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from .stationary import Stationary
from .psi_comp import PSICOMP_RBF, PSICOMP_RBF_GPU
from ...core import Param
from paramz.transformations import Logexp

class RBF(Stationary):
    """
    Radial Basis Function kernel, aka squared-exponential, exponentiated quadratic or Gaussian kernel:

    .. math::

       k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r^2 \\bigg)

    """
    _support_GPU = True
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='rbf', useGPU=False, inv_l=False):
        super(RBF, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name, useGPU=useGPU)
        if self.useGPU:
            self.psicomp = PSICOMP_RBF_GPU()
        else:
            self.psicomp = PSICOMP_RBF()
        self.use_invLengthscale = inv_l
        if inv_l:
            self.unlink_parameter(self.lengthscale)
            self.inv_l = Param('inv_lengthscale',1./self.lengthscale**2, Logexp())
            self.link_parameter(self.inv_l)

    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r**2)

    def dK_dr(self, r):
        return -r*self.K_of_r(r)

    def dK2_drdr(self, r):
        return (r**2-1)*self.K_of_r(r)

    def dK2_drdr_diag(self):
        return -self.variance # as the diagonal of r is always filled with zeros
    def __getstate__(self):
        dc = super(RBF, self).__getstate__()
        if self.useGPU:
            dc['psicomp'] = PSICOMP_RBF()
            dc['useGPU'] = False
        return dc

    def __setstate__(self, state):
        self.use_invLengthscale = False
        return super(RBF, self).__setstate__(state)

    def spectrum(self, omega):
        assert self.input_dim == 1 #TODO: higher dim spectra?
        return self.variance*np.sqrt(2*np.pi)*self.lengthscale*np.exp(-self.lengthscale*2*omega**2/2)

    def parameters_changed(self):
        if self.use_invLengthscale: self.lengthscale[:] = 1./np.sqrt(self.inv_l+1e-200)
        super(RBF,self).parameters_changed()

    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[0]

    def psi1(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[1]

    def psi2(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=False)[2]

    def psi2n(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=True)[2]

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        dL_dvar, dL_dlengscale = self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[:2]
        self.variance.gradient = dL_dvar
        self.lengthscale.gradient = dL_dlengscale
        if self.use_invLengthscale:
            self.inv_l.gradient = dL_dlengscale*(self.lengthscale**3/-2.)

    def gradients_Z_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[2]

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[3:]

    def update_gradients_diag(self, dL_dKdiag, X):
        super(RBF,self).update_gradients_diag(dL_dKdiag, X)
        if self.use_invLengthscale: self.inv_l.gradient =self.lengthscale.gradient*(self.lengthscale**3/-2.)

    def update_gradients_full(self, dL_dK, X, X2=None):
        super(RBF,self).update_gradients_full(dL_dK, X, X2)
        if self.use_invLengthscale: self.inv_l.gradient =self.lengthscale.gradient*(self.lengthscale**3/-2.)

    def dK_dTheta(self,X,X2=None):
        
        local_grad_dict =dict()
    
        local_grad_dict["variance"] = self.K(X, X2=X2)/self.variance

        #self.variance.gradient = np.sum(self.K(X, X2)* dL_dK)/self.variance        
        
        dK_dr = self.dK_dr_via_X(X,X2)
        #print(dK_dr)
        
        if self.ARD:
            """
                dr(x,z)dl_j= -((x_j-zj)^2/r) * (1/l_j^3)
            """
            if X2 is None: X2 = X
            local_grad_dict["lengthscale"] = [None]*self.input_dim
            k_of_r_via_X = self.K_of_r(self._scaled_dist(X,X2))
            tmp = dK_dr*self._inv_dist(X, X2)
            test = [-tmp * np.square(self._unscaled_dist(X[:,q:q+1],X2[:,q:q+1]))/self.lengthscale[q]**3 for q in range(self.input_dim)]
            local_grad_dict["lengthscale"] = test
            """
            for ii in range(self.input_dim):
                
                X_ii = np.reshape(X[:,ii],(-1,1))
                X2_ii = np.reshape(X2[:,ii],(-1,1))
                
                grads = k_of_r_via_X*np.square(self._unscaled_dist(X_ii,X2=X2_ii))/(self.lengthscale[ii]**3)
                print(grads-test[ii])
                local_grad_dict["lengthscale"][ii] = grads
            """
        else:
            local_grad_dict["lengthscale"] = -dK_dr*self._scaled_dist(X,X2)/self.lengthscale
            
        return local_grad_dict
