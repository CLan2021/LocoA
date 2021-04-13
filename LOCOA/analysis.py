#!/usr/bin/env/python

"""
Run analyses including Random Forest Classifier model prediction and Chi-square test.
"""

import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import OPTICS, DBSCAN, KMeans

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from scipy.stats import chisquare


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
        
        self.Analysis = Analysis
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