# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 09:14:49 2016

@author: student
"""

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.neighbors import DistanceMetric



class Interpolator(object):
    """
    Base class for all interpolation methods 
    
    Interpolation is done between the known data Z and the interpolant X.
    Instead of returning the new y value, we are mostly interested
    in a weight matrix W of size len(X[:,0]) x len(Z[:,0]) 
    """
    def __init__(self,Z=None,Y = None):
        """
        
        
        """
        
        self.Z = Z
        
        if not Y is None:
            self.Y = Y
            
    def getWeights(self,X):
        raise NotImplementedError
        
        
    def update_Z(self,Z):
        raise NotImplementedError
        
        
    def dW_dX(self,X):
        raise NotImplementedError
        
        
class NNRegression(Interpolator):
    """
    
    
    """
    
    def __init__(self,Z=None,Y=None,n_neighbors = -1, NNmethod = 'ball_tree',gamma = 1,distance_type = 'rbf'):
        """
        """
        super(NNRegression,self).__init__(Z=Z,Y=Y)
        
        ndims = len(Z[0,:])
        
        if distance_type == 'rbf':
            self.weight_calc = self._rbf
        else:
            print("Unknown distance calculator for NNRegression, {}. Using rbf".format(distance_type))
            self.weight_calc = self._rbf
            
        self.n_neighbors = n_neighbors
        
        if n_neighbors == -1:
            self.n_neighbors = 2^ndims
            
        
        nn_finder = NearestNeighbors(n_neighbors=self.n_neighbors,algorithm=NNmethod)
        nn_finder.fit(Z)
        
        self.nn_finder = nn_finder
        self.gamma = gamma
        self.M = len(Z[:,0])
    def getWeights(self,X):
        """
        First, find nearest neighbors, then get weights via RBF-interpolation.
        
        
        """
        n_data = len(X[:,0])
        n_grid = len(self.Z[:,0])

        distances, indices = self.nn_finder.kneighbors(X)

        row_ind = np.reshape(np.matlib.repmat(np.matrix(np.arange(n_data)).T,1,self.n_neighbors),(n_data*self.n_neighbors,1),order = 'A') 

        #the row indices of the weight matrix W
        column_ind = np.matrix(np.reshape(indices,(n_data*self.n_neighbors,))).T

        #get the weights from the weight function in the right format
        
        weights = self.weight_calc(distances)
        
        row_wise_sum = np.matrix(np.sum(weights,axis=1)).T

        weights = weights / (np.matlib.repmat(row_wise_sum,1,self.n_neighbors))
        
 
        weights_longvec = np.reshape(weights,(n_data*self.n_neighbors,1))
        

        #populate the weight matrix
        W = coo_matrix((np.ravel(weights_longvec),(np.ravel(row_ind),np.ravel(column_ind))),shape=(n_data,n_grid))

        return W
        
    def update_Z(self,Z):
        """
        Update the nearest neighbor method with a new Z
        
        """
        self.Z = Z
        self.nn_finder.fit(Z)
        
        
    def dW_dX(self,X,feature_dims = None):
        """
        Returns a list of matrices where each list item is the partial
        derivative of the weight matrix w.r.t. one of the feature dims 
        specified in "feature_dims" (or all matrix derivatives if feature_dims == None)
        """

        n_data,n_features = np.shape(X)
        
        if feature_dims is None:
            feature_dims = np.arange(n_features)
        
        
        n_grid = len(self.Z[:,0])

        distances, indices = self.nn_finder.kneighbors(X)
        
        indices_long = np.reshape(indices,(n_data*self.n_neighbors,))
        distances_long = np.reshape(distances,(n_data*self.n_neighbors,1))
        neighbors_Z = self.Z[indices_long,:]
        
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
        
        return np.sqrt((x1-x2)**2)
        
        
        
        