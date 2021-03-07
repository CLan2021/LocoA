#!/usr/bin/env python

"""
Install LocoA package. To install locally use:
    'pip install -e .'
"""

from setuptools import setup

setup(
    name="LocoA",
    version="0.0.1",
    author="Catherine Lan",
    author_email="yl4289@columbia.edu",
    url="https://github.com/CLan2021/LocoA",
    description="A package for analyzing locomotor activity data",
    classifiers=["Programming Language :: Python :: 3"],
    entry_points={ 
        #'console_scripts':["module = directory.__main__:main"]
    },
)
