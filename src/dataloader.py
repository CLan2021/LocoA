#!/usr/bin/env/python

"""
Data loading
"""

import os
import pandas as pd
import numpy as np


class Dataloader:
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
            #monitor_cleaned_smooth['hour'] = monitor_cleaned_smooth['time'].str[:2].astype(int) #getting hour (first two characters) and setting as integer
            #use list comprehension to add a 'timeofday' column that designates a row as 'night' if 7pm-4am, otherwise assigns as 'day'
            #monitor_cleaned_smooth['timeofday'] = ['night' if x > 19 & x < 4 else 'day' for x in monitor_cleaned_smooth['hour']] 

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
