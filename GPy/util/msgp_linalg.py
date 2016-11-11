# -*- coding: utf-8 -*-

#
#   The utility functions for usage in grid_gaussian_inference and kern_grid
#
import numpy as np
import scipy 
import warnings
from scipy import linalg

def solveMVM(p,mvm_func,shape_mvm,cgtol = 1e-7,cgmit=700):
    mvm_operator = scipy.sparse.linalg.LinearOperator(shape_mvm,mvm_func, dtype='float64')
    q,flag = scipy.sparse.linalg.cg(mvm_operator,b=p,tol = cgtol,maxiter = cgmit)
#q,flag = conjgrad(mvm,p,tol=cgtol,maxit=cgmit)
    if not flag == 0:
        warnings.warn('CG not converged after {} iterations;'.format(cgmit))
    return q    
    
def mvm_K(p,K,sn2,M):
    """
    Calc [W*K*W' + I]*p using kronecker strucutre in K
    """

    return M.dot(kronmvm(K,(M.T).dot(p)))+sn2*p
    

def kronmvm(As,x,transp = False,x_sparse=False):
    """
    
    TO-DO: Need to finish this
    TO-DO: No toeplitz yet
    TO-DO: Understand this O.o
    """

    is_1d_vec = False
    if len(np.shape(x)) == 1: ##<- check if its a (n,) vector -> is this a good way?
        x = np.reshape(x,(-1,1))

        is_1d_vec = True
        
    if not transp is None and transp:
        for i in range(len(As)):
            As[i] = As[i].T
            
    for i in range(len(As)):
        As[i] = np.asarray(As[i])
    m = np.zeros((len(As),1))
    n = np.zeros((len(As),1))
    
    for i in range(len(n)):
        m[i],n[i] = np.shape(As[i])
        
    d = np.shape(x)[1]
    b=x
    for ii in range(len(n)):
        a = np.reshape(b,(np.prod(m[0:ii]),n[ii],np.prod(m[ii+1:len(n)])*d))
        tmp = np.dot(np.reshape(a.transpose([0,2,1]),(-1,n[ii])),As[ii].T)
        b = (np.reshape(tmp,(np.shape(a)[0],np.shape(a)[2],m[ii]))).transpose([0,2,1])

    b = np.reshape(b,(np.prod(m),d))

    if is_1d_vec:
        b = np.ravel(b)
    return b
"""   
def kronmvm_test(As,x,transp = False,x_sparse=False):
    
    
    TO-DO: Need to finish this
    TO-DO: No toeplitz yet
    TO-DO: Understand this O.o
    

    is_1d_vec = False
    if len(np.shape(x)) == 1: ##<- check if its a (n,) vector -> is this a good way?
        x = np.reshape(x,(-1,1))

        is_1d_vec = True
        
    if not transp is None and transp:
        for i in range(len(As)):
            As[i] = As[i].T
    m = np.zeros((len(As),1))
    n = np.zeros((len(As),1))
    
    for i in range(len(n)):
        m[i],n[i] = np.shape(As[i])
        
    d = np.shape(x)[1]
    b=x
    for ii in range(len(n)):
     
        #a = np.reshape(b,(2,5,300))

        a = np.reshape(b,(np.prod(m[0:ii]),n[ii],np.prod(m[ii+1:len(n)])*d))
        tmp = np.reshape(a.transpose([0,2,1]),(-1,n[ii])).dot(As[ii].T)
        b = (np.reshape(tmp,(np.shape(a)[0],np.shape(a)[2],m[i]))).transpose([0,2,1])

    b = np.reshape(b,(np.prod(m),d))

    if is_1d_vec:
        b = np.ravel(b)
    return b
"""
def sparse_reshape(a, shape):
    """Reshape the sparse matrix `a`.

    Returns a coo_matrix with shape `shape`.
    """
    if not hasattr(shape, '__len__') or len(shape) != 2:
        raise ValueError('`shape` must be a sequence of two integers')

    c = a.tocoo()
    nrows, ncols = c.shape
    size = nrows * ncols

    new_size =  shape[0] * shape[1]
    if new_size != size:
        raise ValueError('total size of new array must be unchanged')

    flat_indices = ncols * c.row + c.col
    new_row, new_col = divmod(flat_indices, shape[1])

    b = coo_matrix((c.data, (new_row, new_col)), shape=shape)
    return b


def eigr(A,tol = None):
    """

    """
    
    D,V = linalg.eigh((A+A.T)/2)
    n = np.shape(A)[0]

    #sort_idx = D.argsort()[::-1][:n]
    

    #d = np.maximum(np.ndarray.real(np.diag(D)),0) -> already in vector form
    
    sort_index= np.argsort(D)[::-1]
    d = D[sort_index]

    #d,order = np.sort(D,order='descend')
    if tol is None:
        tol = n*np.spacing(np.max(np.real(d)))
        #tol = n*np.spacing(np.max(d)) <- not working
        
    r = np.sum([d>tol])
    d[r:n] = 0
    D = np.diag(d)

    ##TO-DO: aren't we overriting sth. here?
    V_tmp = V

    V[:,0:r] = np.real(V_tmp[:,sort_index[0:r]])

    V[:,r:n] = null_space(V[:,0:r].T,eps = tol)
    ##dafuq
    return V,D
    
##line 149 in infGrid.m
def null_space(A, eps=1e-15):
    """
    Return a orthogonal basis of the nullspace of the matrix A via svd
    
    """
    u, s, vh = np.linalg.svd(A)
    n = A.shape[1]   # the number of columns of A
    if len(s)<n:
        expanded_s = np.zeros(n, dtype = s.dtype)
        expanded_s[:len(s)] = s
        s = expanded_s
    null_mask = (s <= eps)
    null_space = np.compress(null_mask, vh, axis=0)
    
    return np.transpose(null_space)
    