# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 09:14:49 2016

@author: student
"""

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.neighbors import DistanceMetric
from . import Timer
from numpy import matlib
import itertools
from sklearn.utils.extmath import cartesian


class Interpolator(object):
    """
    Base class for all interpolation methods 
    
    Interpolation is done between the known data Z and the interpolant X.
    Instead of returning the new y value, we are mostly interested
    in a weight matrix W of size len(X[:,0]) x len(Z[:,0]) 
    """
    def __init__(self,grid_dims, Z_all=None,Z=None,Y = None):
        """
        
        
        """
        self.grid_dims = grid_dims
        self.Z_all = Z_all
        self.Z = Z
        
        if not Y is None:
            self.Y = Y
            
    def getWeights(self,X):
        raise NotImplementedError
        
        
    def update_Z(self,Z,Z_all):
        raise NotImplementedError
        
        
    def dW_dX(self,X):
        raise NotImplementedError
        
        
class NNRegression(Interpolator):
    """
    
    
    """
    
    def __init__(self,grid_dims,Z_all=None,Z=None,Y=None,n_neighbors = -1, NNmethod = 'ball_tree',gamma = 1,distance_type = 'rbf'):
        """
        """
        super(NNRegression,self).__init__(grid_dims,Z_all = Z_all,Z=Z,Y=Y)
        
        ndims = len(Z_all[0,:])
        
        if distance_type == 'rbf':
            self.weight_calc = self._rbf
        else:
            print("Unknown distance calculator for NNRegression, {}. Using rbf".format(distance_type))
            self.weight_calc = self._rbf
            
        self.n_neighbors = n_neighbors
        
        if n_neighbors == -1:
            self.n_neighbors = 3**ndims
            
        
        nn_finder = NearestNeighbors(n_neighbors=self.n_neighbors,algorithm=NNmethod)
        nn_finder.fit(Z_all)
        
        
        
               
        
        
        
        self.nn_finder = nn_finder
        self.gamma = gamma
        
        
        """
        Test the per_dim distance with NN algo
        
        """ 
        
        self.nn_finders = [None]*len(Z)
        for ii in range(len(Z)):
            nn_finder = NearestNeighbors(n_neighbors=3,algorithm=NNmethod)
            nn_finder.fit(Z[ii])
            self.nn_finders[ii] = nn_finder
            
        
    def _grid_find_neighbors(self,X,k=3):
        n_x = np.shape(X)[0]
        
        n_dims = len(self.grid_dims)
        
        n_neighbors = k**n_dims        
        
        neighbors_per_dim_X,distances_per_dim  = self._find_k_nearest_per_dim(X,k=k)
       
        
        
        
        #for ii in range(n_x):
            #with Timer() as t:
        #distances_per_grid_dim, neighbor_coordinates = np.reshape(neighbors_per_dim_X[ii,0,:],(-1,1))
        #indices_all = np.zeros((n_x*n_neighbors,))
        distances_all = np.zeros((n_x,n_neighbors,n_dims))
        neighbor_coordinates_all = np.zeros((n_x*n_neighbors,n_dims))
          
        for ii in range(n_x):
            
            neighbor_coordinates_all[ii*n_neighbors:(ii+1)*n_neighbors,:] = cartesian(neighbors_per_dim_X[ii,:,:])
            distances_all[ii,:,:] = cartesian(distances_per_dim[ii,:,:])
            
        indices_all = self._kron_grid_indices_to_matrix_indices(neighbor_coordinates_all.astype(int))

        
        distances_all = np.mean(np.square(distances_all),axis = 2)
        

        return distances_all,indices_all
        
    def _dist_to_grid(self,x,grid_coordinates):
        """
        
        input:
            x:
            grid_coordinates:
            
        """
        if len(np.shape(x)) > 1:
            x = np.ravel(x)
        n_dim = len(x)
        n_z = np.shape(grid_coordinates)[0]
        
        n_grid_dims = len(self.grid_dims)
        
        dist = np.zeros((n_z,1))
        for ii in range(n_z):
            Z_full = np.zeros((n_dim,1))
            for jj in range(n_grid_dims):
                Z_full[self.grid_dims[jj],0]= self.Z[jj][grid_coordinates[ii,jj],:]

            dist[ii,0] = self._eucl_dist(np.reshape(x,(1,-1)),Z_full.T)
        
        return dist
        
    
    def _find_k_smallest_list(self,input_array,k):
        
        # partition, slice back to the k smallest elements, convert back to a Python list
        
        k_smallest_dims = np.argpartition(input_array, k)[:k]
       
         
        return k_smallest_dims
    
    
    def _find_k_nearest_per_dim(self,X,k=3):
        n_pred =np.shape(X)[0]
        n_dims = len(self.grid_dims)
        
        neighbor_indices = np.zeros((n_pred,n_dims,k))
        dist_per_dim = np.zeros((n_pred,n_dims,k))
        for ii in range(n_dims):
            dist_dim_ii, neighbor_indices_ii = self.nn_finders[ii].kneighbors(X[:,self.grid_dims[ii]])
            neighbor_indices[:,ii,:] = neighbor_indices_ii
            dist_per_dim[:,ii,:] = dist_dim_ii
            
        return neighbor_indices, dist_per_dim
        
        
        """
        n_pred =np.shape(X)[0]
        n_dims = len(self.grid_dims)
        neighbor_indices = np.zeros((n_pred,n_dims,k))
        for ii in range(n_dims):
            
            for jj in range(n_pred):
             
                dist_dim_ii = self._eucl_dist(self.Z[ii],X[jj,self.grid_dims[ii]])

                ind_k_closest = self._find_k_smallest_list(dist_dim_ii,k)

                neighbor_indices[jj,ii,:] = np.array(ind_k_closest)
            
                
        return neighbor_indices
        """
        
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
    
    def _kron_grid_indices_to_matrix_indices(self,indices):
        """
        Compute the position (index) along an axis in the full kronecker matrix,
        given indices of the single grid dimensions
        
        input: 
            indices: array of size n_data x n_grid_dims containing for each gridpoint
                     the grid coordinates
                     
                     
        TO-DO: we can make this faster when pre-computing the "prod" term
        """
        
        n_data, n_grid_dims = np.shape(indices)
        
        n_data_grid_dims = []
        for jj in range(n_grid_dims):
            n_data_grid_dims.append(np.shape(self.Z[jj])[0])
        
        kron_indices = [0]*n_data
        
        for jj in range(n_grid_dims-1):
            kron_indices = kron_indices + indices[:,jj]*np.prod(n_data_grid_dims[jj+1:])
        
        """        
        for ii in range(n_data):
            ind = 0
            for jj in range(n_grid_dims-1):
                ind = ind + indices[ii,jj]*np.prod(n_data_grid_dims[jj+1:])
            ind = ind + indices[ii,n_grid_dims-1]
            kron_indices[ii] = ind
        """
                
        return kron_indices
        
            
    def getWeights(self,X):
        """
        First, find nearest neighbors, then get weights via RBF-interpolation.
        
        
        """
        n_data = len(X[:,0])
        n_grid = len(self.Z_all[:,0])
        
        """
        with Timer() as t:
            distances, indices = self.nn_finder.kneighbors(X)
        print("Runtime KD: {} s".format(t.secs))
        
        row_ind = np.reshape(np.matlib.repmat(np.matrix(np.arange(n_data)).T,1,self.n_neighbors),(n_data*self.n_neighbors,1),order = 'A') 

        #the row indices of the weight matrix W
        column_ind = np.matrix(np.reshape(indices,(n_data*self.n_neighbors,))).T

        #get the weights from the weight function in the right format
        
        weights = self.weight_calc(distances)
        
        row_wise_sum = np.matrix(np.sum(weights,axis=1)).T

        weights = weights / (np.matlib.repmat(row_wise_sum,1,self.n_neighbors))
        
 
        weights_longvec = np.reshape(weights,(n_data*self.n_neighbors,1))
        """
        

        
        with Timer() as t:
            distances,column_ind  = self._grid_find_neighbors(X)
        print("Runtime grid neighbors: {} s".format(t.secs))
            #print(np.shape(distances))
        #print(np.shape(indices))
        
        with Timer() as t:

            distances_test, indices_test = self.nn_finder.kneighbors(X)
        print("Runtime KD: {}".format(t.secs))
        print(distances_test)
        
        column_ind = np.reshape(indices_test,(n_data*self.n_neighbors,))
    
        distances = distances_test
        
        
        row_ind = np.reshape(np.matlib.repmat(np.matrix(np.arange(n_data)).T,1,self.n_neighbors),(n_data*self.n_neighbors,1),order = 'A') 

        weights = self.weight_calc(distances)
        row_wise_sum = np.matrix(np.sum(weights,axis=1)).T

        weights = weights / (np.matlib.repmat(row_wise_sum,1,self.n_neighbors))
        #populate the weight matrix
        weights_longvec = np.reshape(weights,(n_data*self.n_neighbors,1))
    
    
    
    
    
        W = coo_matrix((np.ravel(weights_longvec),(np.ravel(row_ind),np.ravel(column_ind))),shape=(n_data,n_grid))
    
        return W
        
    def update_Z(self,Z,Z_all):
        """
        Update the nearest neighbor method with a new Z
        
        """
        self.Z = Z
        self.Z_all = Z_all
        self.nn_finder.fit(Z)
        
        
    def dW_dX(self,X,feature_dims = None, compute_W = False):
        """
        Returns a list of matrices where each list item is the partial
        derivative of the weight matrix w.r.t. one of the feature dims 
        specified in "feature_dims" (or all matrix derivatives if feature_dims == None)
        """

        n_data,n_features = np.shape(X)
        
        if feature_dims is None:
            feature_dims = np.arange(n_features)
        
        
        n_grid = len(self.Z_all[:,0])
        
        with Timer() as t:

            distances_test, indices_test = self.nn_finder.kneighbors(X)
        print("Runtime KD: {}".format(t.secs))
        print(distances_test)
        
        
        indices_long = np.reshape(indices_test,(n_data*self.n_neighbors,))
        distances = distances_test
        #with Timer() as t:

        #    distances, indices_long = self._grid_find_neighbors(X)
        #print("Runtime grid search: {}".format(t.secs))
        #print(distances)
        #print(indices_long-indices_long_test)

        distances_long = np.reshape(distances,(n_data*self.n_neighbors,1))
        neighbors_Z = self.Z_all[indices_long,:]
        
        weights = self.weight_calc(distances)
        
        row_wise_sum = np.matrix(np.sum(weights,axis=1)).T

        X_longvec = np.reshape(np.matlib.repmat(np.matrix(X),1,self.n_neighbors),(n_data*self.n_neighbors,n_features),order = 'A')
        
        drbfdx2 = self._drbf_dx2(neighbors_Z,X_longvec,distances_long,feature_dims = feature_dims)
        
        row_wise_sum_long = np.reshape(np.matlib.repmat(np.matrix(row_wise_sum),1,self.n_neighbors),(n_data*self.n_neighbors,1),order = 'A')
        
        weights_long = np.reshape(weights,(n_data*self.n_neighbors,1))
        
        
        
        sum_drbf_dx2_per_neighborhood = np.zeros((n_data*self.n_neighbors,1))
        for ii in range(0,n_data*self.n_neighbors,self.n_neighbors):
            sum_drbf_dx2_per_neighborhood[ii:ii+self.n_neighbors] = np.sum(drbfdx2[ii:ii+self.n_neighbors,:])
            
        

        #get the weights from the weight function in the right format
        
        dWdX = (np.multiply(drbfdx2,row_wise_sum_long) - np.multiply(weights_long,sum_drbf_dx2_per_neighborhood))

        dWdX = dWdX /(np.square(row_wise_sum_long))
        #the row indices of the weight matrix W
        row_ind = np.reshape(np.matlib.repmat(np.matrix(np.arange(n_data)).T,1,self.n_neighbors),(n_data*self.n_neighbors,1),order = 'A') 


        column_ind = np.matrix(indices_long).T
       
        dWdX_list = list()
        for ii in range(len(feature_dims)):
            dWdX_list.append(coo_matrix((np.ravel(dWdX[:,ii]),(np.ravel(row_ind),np.ravel(column_ind))),shape=(n_data,n_grid)))
            
        if compute_W:
            
            
            weights_long_norm = weights_long / row_wise_sum_long
            
            W = coo_matrix((np.ravel(weights_long_norm),(np.ravel(row_ind),np.ravel(column_ind))),shape=(n_data,n_grid))
            
            return dWdX_list, W
            
            

        return dWdX_list 
    def _rbf(self,dist):
        """
        """
        return np.exp(-self.gamma*dist)
        
    def _drbf_dx2(self,x1,x2,distances,feature_dims = None):
        """
        Returns a matrix of shape n_data,n_features
        where entry i,j is the partial derivative of 
        rbf(x_i,x2) w.r.t. x2_j
        
        Input:
            x2: data-matrix of shape Nxn_features
            x1: vector of shape 1 x n_features or array of length n_features
        """
        return -self.gamma * np.multiply(np.multiply(self._rbf(distances) , (1./distances)),(x1[:,feature_dims]-x2[:,feature_dims]))
        
    def _eucl_dist(self,x1,x2):
        """
        
        """
        return np.linalg.norm(x1-x2,axis=1)
        
        
        
        