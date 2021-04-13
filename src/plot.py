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
    def __init__(self, Analysis):            
    
    def plot_setup(self):
        '''
        Run three functions in a row to set up class instance for plotting.
        '''
        
        self.Analysis.get_dr()
        self.Analysis.get_pwdists()
        self.Analysis.get_group_idxs()
        
    
    def dbscan_clustering(self):
        '''
        DBSCAN clustering analysis.
        '''
        
        dr_wg = self.Analysis.dr[self.Analysis.group_idxs_ > 0]
        gcolor_map = self.Analysis.group_idxs / (self.Analysis.group_idxs.max() + 1)
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        axs[0][0].scatter(dr_wg[:,0], dr_wg[:,1], c=gcolor_map)
        axs[0][1].scatter(dr_wg[:,2], dr_wg[:,1], c=gcolor_map)
        axs[1][0].scatter(dr_wg[:,0], dr_wg[:,2], c=gcolor_map)
        for tid in range(dr_wg.shape[0]):
            axs[0][0].text(dr_wg[tid,0], dr_wg[tid,1], self.Analysis.group_idxs[tid])
            axs[0][1].text(dr_wg[tid,2], dr_wg[tid,1], self.Analysis.group_idxs[tid])
            axs[1][0].text(dr_wg[tid,0], dr_wg[tid,2], self.Analysis.group_idxs[tid])
        plt.show()
        
        
    def line_graph(self):
        '''
        Line graph analysis.
        '''
        
        plt.figure(figsize=(20,10))
        plt.plot(self.Analysis.act_origs_npy[self.Analysis.group_idxs_ == 0].mean(axis=0)) # blue
        plt.plot(self.Analysis.act_origs_npy[self.Analysis.group_idxs_ == 1].mean(axis=0)) # orange
        plt.plot(self.Analysis.act_origs_npy[self.Analysis.group_idxs_ == 2].mean(axis=0)) # green
        plt.axvline(1439.33, linestyle='--')
        plt.axvline(1439.33*2, linestyle='--')
        plt.axvline(1439.33*3, linestyle='--')