#!/usr/bin/env python

'''
LocoA code reorganized into class objects.
'''

from sys import call_tracing
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS, SpectralEmbedding, TSNE, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import OPTICS, DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
import umap
import mpl_toolkits.mplot3d.axes3d as p3
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from scipy.stats import chisquare


class Simulator:
    '''
    A class object to load LAM data and format CSV files.
    '''
    
    # Initialize class instance.  All params are defaults and can be changed.
    def __init__(
        self,
        use_log = True,
        use_std = True,
        ignored = 1440,
        group_func = 'mean', # Alt: 'mean_and_median'
        nn = 50,
        reducer_name = 'umap', # Alt: 'pca'
        custom_profile = False
        ):
        
        # Store arguments [what does each one do?]
        self.use_log = use_log
        self.use_std = use_std
        self.ignored = ignored
        self.group_func = group_func
        self.nn = nn
        self.reducer_name = reducer_name
        
        # Short if-else statement to control custom profile param.
        if not custom_profile:
            self.custom_profile = 'nocustom'
        else:
            self.custom_profile = 'cosine-nn%d' % self.nn
        
        # Load and store metadata.
        self.one_meta2 = pd.read_csv('./data/one_meta2.csv', sep='\t')
        
        # Create list objects to be filled with data later [what is each one?]
        self.act_digests = []
        self.act_origs = []
        self.metas = []
        self.monitor_cleaned_smooths = []
        
        
    def make_profile(self):
        '''
        Build profile from params.
        '''
        
        # Build profile string.  (What is this used for?)
        profile = '_'.join([self.reducer_name, 'log' if self.use_log else 'nolog', self.group_func, 
                            'std' if self.use_std else 'nostd', 'ignore%d' % self.ignored, 
                            self.custom_profile])
        return profile
    
    
    def format_data(self, win_size = 10, conv_size = 3):
        '''
        Format data for downstream analysis.
        '''
        
        # Clear summary lists.  This prevents accumulating duplicate results if the function is run more than once.
        self.act_digests.clear()
        self.act_origs.clear()
        self.metas.clear()
        self.monitor_cleaned_smooths.clear()
        
        # Load text files.  The user is expected to have placed them appropriately before running this function.
        txts = [f for f in os.listdir('./data') if f.endswith('.txt')] #List all the text files within a directory
        
        # For-loop to process each text file.
        for txt in txts: 
            meta = self.one_meta2[self.one_meta2.File_Name == os.path.splitext(txt)[0]] #getting metadata from csv
            if len(meta) == 0: #does this say if no metadata available, just print the text file as is?
                print(txt)
                continue
            
            # Loading files and creating date, time, and cleaning number of columns/rows
            monitor = pd.read_csv('./data/%s' % txt, sep='\t', header=None) #load txt file as csv
            monitor = monitor.rename({1:'date', 2:'time'}, axis=1) #rename columns as date and time
            monitor_cleaned = pd.concat([monitor.iloc[:,1:3], monitor.iloc[:,10:]], axis=1) #concatenate row number, date, time, and from 10th beetle on 
            monitor_cleaned = monitor_cleaned.iloc[self.ignored:,:] #slice out problematic row 1440 I'm assuming?
            
            # Adding columns and convolution
            monitor_cleaned_smooth = monitor_cleaned.iloc[:,2:].apply(np.convolve, v=np.ones(conv_size), mode='valid') #convolution transformation for dates?
            if self.use_log: #if log-scale
                monitor_cleaned_smooth = np.log(monitor_cleaned_smooth + 1) #log+1 to avoid issues with log-scale I'm assuming?

            monitor_cleaned_smooth = pd.concat([monitor_cleaned.iloc[(conv_size-1):,:2].reset_index(drop=True), 
                                                monitor_cleaned_smooth], axis=1) #cleaning up convolution
            hms = np.array([t.replace(' ', ':').split(':') for t in monitor_cleaned_smooth.time], dtype=int)
            monitor_cleaned_smooth['h'] = hms[:,0] #creating "h" column
            monitor_cleaned_smooth['mNcell'] = hms[:,1] // win_size #creating 'mNcell' column 
            
            # Creating night-time vs. day-time column
            monitor_cleaned_smooth['hour'] = monitor_cleaned_smooth['time'].astype(int) #converting hours to integers for easy classification
            #use list comprehension to add a 'timeofday' column that designates a row as 'night' if after 7pm, otherwise assigns as 'day'
            monitor_cleaned_smooth['timeofday'] = ['night' if x > 19 else 'day' for x in monitor_cleaned_smooth['hour']] 

            # Either calculate only mean, or both mean and median.
            if self.group_func == 'mean':
                act_digest = pd.concat([monitor_cleaned_smooth.groupby(['h', 'mNcell']).mean().T, 
                                        monitor_cleaned_smooth.groupby(['h', 'mNcell']).std().T], axis=1) #gets mean and standard deviation for 'h' and 'mNcell' columns
            else:
                q1 = monitor_cleaned_smooth.groupby(['h', 'mNcell']).apply(pd.DataFrame.quantile, q=.25).T.iloc[:-2] #calculate bottom quartile for 'h' and 'mNcell'
                q3 = monitor_cleaned_smooth.groupby(['h', 'mNcell']).apply(pd.DataFrame.quantile, q=.75).T.iloc[:-2] #calculate top quartile
                
                monitor_cleaned_smooth_min = monitor_cleaned_smooth.groupby(['h', 'mNcell']).min().T.iloc[2:] #assign minimum values for 'h' and 'mNcell'
                monitor_cleaned_smooth_max = monitor_cleaned_smooth.groupby(['h', 'mNcell']).max().T.iloc[2:] #assign maximum values
                IQR = q3 - q1 #Interquartile range defined
                monitor_cleaned_smooth_whisker_min = q1 - 1.5 * IQR #Creating CIs
                monitor_cleaned_smooth_whisker_max = q3 + 1.5 * IQR #Creating CIs
        
                whisker_min_oob = (monitor_cleaned_smooth_whisker_min < monitor_cleaned_smooth_min) #if minimum of CI is smaller than observed minimum
                whisker_max_oob = (monitor_cleaned_smooth_whisker_max > monitor_cleaned_smooth_max) #same but for maximum CI compared to observ
                monitor_cleaned_smooth_whisker_min[whisker_min_oob] = monitor_cleaned_smooth_min[whisker_min_oob] 
                monitor_cleaned_smooth_whisker_max[whisker_max_oob] = monitor_cleaned_smooth_max[whisker_max_oob]

                #creating new dataframe called act_digest concatenating grouped summary statistics and CI information on 'h' and 'mNcell' 
                act_digest = pd.concat([
                monitor_cleaned_smooth.groupby(['h', 'mNcell']).mean().T,  #group by 'h' and 'mNcell' and output mean
                monitor_cleaned_smooth.groupby(['h', 'mNcell']).std().T, #standard deviation
                monitor_cleaned_smooth.groupby(['h', 'mNcell']).median().T,  #median
                q1, #lower quartile
                q3, #upper quartile
                monitor_cleaned_smooth_whisker_min, #CI lower range
                monitor_cleaned_smooth_whisker_max, #CI upper range
                ], axis=1)
            
            #Subset array of monitor_cleaned_smooth dataframe calling it act_orig
            act_orig = monitor_cleaned_smooth.iloc[:,2:34].T
            
            # Append results to lists.
            self.act_digests.append(act_digest)
            self.act_origs.append(act_orig)
            self.metas.append(meta)
            self.monitor_cleaned_smooths.append(monitor_cleaned_smooth)
            
        # Concatenate and return results for downstream analysis.
        meta_union = pd.concat(self.metas).reset_index(drop=True)
        meta_sum = np.sum(meta_union.Instar.isna())
        act_digests_npy = np.concatenate(self.act_digests)
        act_origs_npy = np.concatenate(self.act_origs)
        monitor_cleaned_smooths_union = pd.concat(self.monitor_cleaned_smooths).reset_index(drop=True)
        return meta_union, meta_sum, act_digests_npy, act_origs_npy, monitor_cleaned_smooths_union


class Analysis:
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
        
    
    def scatter3d(self, data3d, color_map=None, azimuth=-74, elevation=54, figsize=(10, 10)):
        '''
        Build 3D scatterplot.  (Currently doesn't link to anything else?)
        '''

        fig = plt.figure(figsize=figsize)
        ax = p3.Axes3D(fig)
        ax.view_init(elevation, azimuth)

        if color_map is None:
            ax.scatter(data3d[:,0], data3d[:,1], data3d[:,2],
                        s=30, edgecolor='k')
        else:
            ax.scatter(data3d[:,0], data3d[:,1], data3d[:,2],
                        color=plt.cm.jet(color_map),
                        s=30, edgecolor='k')
        return ax
    
    
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
        
        
    def model_training(self):
        '''
        Model training analysis.
        '''
        
        s, y_ = np.unique(self.meta_union.Instar.values, return_inverse=True)
        y_[self.meta_union.Instar.isna()] = -1
        y = y_[y_ != -1]
        
        x_ = self.meta_union[['Source']].copy()
        x_['group'] = self.group_idxs_   
        x = x_[y_ != -1].copy()
        
        rfc = RandomForestClassifier(random_state=42, max_depth=5, n_estimators=1000, criterion='entropy')
        
        enc = OneHotEncoder(sparse=False).fit(x)
        new_x = enc.transform(x)
        x_train, x_test, y_train, y_test = train_test_split(new_x, y, test_size=.5, random_state=42)
        
        # The present Jupyter notebook example calls a lot of attributes without saving them to variables.  Are these
        # desired outputs of the model? Are they meant to be saved and used somewhere else? It's unclear to me how to
        # format the remaining code for this model.
        
        
    def chi_square(self):
        '''
        Chi-square analysis.
        '''
        
        group_sizes = pd.DataFrame({'gid': self.group_idxs}).groupby('gid').size().values
        
        # Expected frequencies in each category.  By default the categories are assumed to be equally likely.
        f_exp = group_sizes / group_sizes.sum()

        # cand_cols = ['Nest', 'Source', 'Elevation', 'Gen', 'Sex', 'Photo', 'Instar']
        cand_cols = ['Instar']

        # Create arrays.
        biased_chars_all = np.array([])
        chi2_all = np.array([])
        pvalue_all = np.array([])
        char_sample_size_all = np.array([])
        cat_col_str_all = np.array([])
        
        # For-loop over columns.
        for i in range(len(cand_cols)):
            for j in range(i, len(cand_cols)):
                cat_col = list(np.unique([cand_cols[i], cand_cols[j]]))
                cat_col_str = '_x_'.join(cat_col)
                # print(cat_col_str)
                self.meta_union[cat_col_str] = self.meta_union[cat_col].replace(np.nan, 'NaN').astype(str).apply(
                    '_x_'.join, axis=1)

                char_to_group = pd.DataFrame.from_dict({'gid': self.group_idxs, 
                                                        'char': self.meta_union[cat_col_str].values[self.group_idxs_ > 0]}
                                                      ).pivot_table(index='gid', columns='char', aggfunc=len)
                char_to_group = char_to_group.replace(np.nan, 0)

                f_exp_weighted = char_to_group.sum().values * np.repeat(
                    np.expand_dims(f_exp, axis=0), char_to_group.shape[1], axis=0).T
        
                chi2test = chisquare(char_to_group, f_exp=f_exp_weighted)
        
                pvalue_thres_idxs = (chi2test.pvalue < 0.05)
                biased_chars = char_to_group.columns.values[pvalue_thres_idxs]
                biased_chars_all = np.append(biased_chars_all, biased_chars)
                chi2 = chi2test.statistic[pvalue_thres_idxs]
                chi2_all = np.append(chi2_all, chi2)
                pvalue_all = np.append(pvalue_all, chi2test.pvalue[pvalue_thres_idxs])
                char_sample_size_all = np.append(char_sample_size_all, char_to_group.sum()[pvalue_thres_idxs].values)
                cat_col_str_all = np.append(cat_col_str_all, np.repeat(cat_col_str, chi2.shape[0]))
                
        return char_to_group, chi2test, pvalue_thres_idxs, biased_chars_all, chi2_all, pvalue_all, char_sample_size_all, cat_col_str_all