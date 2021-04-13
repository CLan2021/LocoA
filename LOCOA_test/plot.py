#!/usr/bin/env/python
"""
Defining fisheye center circle coordinates for future gap fraction calculation
"""


import umap
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import OPTICS, DBSCAN, KMeans
from matplotlib import pyplot as plt

class Plot:
    '''
    A class object to analyze data.
    '''
    
    # Initialize class instance.  Param defaults of None are intended to be replaced with output from Simulator functions.
    # Functions within this class may fail without proper input.
    def __init__(
        self,
        meta_union = None,
        act_digests_npy = None,
        act_origs_npy = None,
        min_samples = 20,
        nn = 50
        ):
        
        self.meta_union = meta_union
        self.act_digests_npy = act_digests_npy
        self.act_origs_npy = act_origs_npy
        self.min_samples = min_samples
        self.nn = nn
        
        # Initialize additional objects to hold calculations for plotting and stats.
        self.dr = None
        self.pwdists = None
        self.group_idxs_ = None
        self.group_idxs = None
        
    
    def stdscaler(self, X, use_std=True):
        '''
        Standardize input data.
        '''
        
        if use_std:
            return StandardScaler().fit_transform(X)
        else:
            return X
        
    
    def get_dr(self):
        '''
        Get dr and save to class instance.
        '''
        reducer = umap.UMAP(n_components=3, metric='cosine', n_neighbors=self.nn, random_state=5566)
        act_digests_npy_transformed = self.stdscaler(self.act_digests_npy, use_std=True)
        dr = reducer.fit_transform(act_digests_npy_transformed)
        self.dr = dr
        
    
    def get_pwdists(self):
        '''
        Get pairwise distances and save to class instance.
        '''
        
        pwdists = pairwise_distances(self.dr)
        self.pwdists = pwdists
    
    
    def get_group_idxs(self):
        '''
        Get group indices and save to class instance.
        '''

        shortest_dists_mean = np.take_along_axis(self.pwdists, 
                                                 np.argsort(self.pwdists)[:,1:(1+self.min_samples)], axis=1).mean(axis=1)
        shortest_dists_mean_std = shortest_dists_mean.std()
        eps = shortest_dists_mean.mean() + 2 * shortest_dists_mean_std
        clusterer = DBSCAN(eps=eps, min_samples=self.min_samples)
        group_idxs_ = clusterer.fit_predict(self.dr) + 1
        group_idxs = group_idxs_[group_idxs_ > 0]
        
        # Save the two variants of group indexes to the class instance.
        self.group_idxs_ = group_idxs_
        self.group_idxs = group_idxs
        
    
    def plot_setup(self):
        '''
        Run three functions in a row to set up class instance for plotting.
        '''
        
        self.get_dr()
        self.get_pwdists()
        self.get_group_idxs()
        
    
    def dbscan_clustering(self):
        '''
        DBSCAN clustering analysis.
        '''
        
        dr_wg = self.dr[self.group_idxs_ > 0]
        gcolor_map = self.group_idxs / (self.group_idxs.max() + 1)
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        axs[0][0].scatter(dr_wg[:,0], dr_wg[:,1], c=gcolor_map)
        axs[0][1].scatter(dr_wg[:,2], dr_wg[:,1], c=gcolor_map)
        axs[1][0].scatter(dr_wg[:,0], dr_wg[:,2], c=gcolor_map)
        for tid in range(dr_wg.shape[0]):
            axs[0][0].text(dr_wg[tid,0], dr_wg[tid,1], self.group_idxs[tid])
            axs[0][1].text(dr_wg[tid,2], dr_wg[tid,1], self.group_idxs[tid])
            axs[1][0].text(dr_wg[tid,0], dr_wg[tid,2], self.group_idxs[tid])
        plt.show()
        
        
    def line_graph(self):
        '''
        Line graph analysis.
        '''
        
        plt.figure(figsize=(20,10))
        plt.plot(self.act_origs_npy[self.group_idxs_ == 0].mean(axis=0)) # blue
        plt.plot(self.act_origs_npy[self.group_idxs_ == 1].mean(axis=0)) # orange
        plt.plot(self.act_origs_npy[self.group_idxs_ == 2].mean(axis=0)) # green
        plt.axvline(1439.33, linestyle='--')
        plt.axvline(1439.33*2, linestyle='--')
        plt.axvline(1439.33*3, linestyle='--')