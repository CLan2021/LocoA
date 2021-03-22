#!/usr/env/bin python

"""
Example functions for reading data
"""

import pandas as pd


def read_loco_file(infile):
    """
    Read the locomotor data file with pandas
    """
    # load with pandas
    data = pd.read_csv(infile, sep='\t', header=None)
    return data


def read_meta_file(infile):
    """
    Read meta info file as pandas dataframe
    """
    data = pd.read_csv(infile, sep='\t')
    return data


def next_function():
    """
    A function to do the next thing... such as draw a line plot
    from the data.
    """
    # ...



if __name__ == "__main__":

    loc_data = read_loco_file("../data/Monitor20200924_0928_soil.txt")
    meta_data = read_meta_file("../data/one_meta2.csv")

    print(loc_data.head())

    print(meta_data.columns)
    print(meta_data.head())
