# Comments and quetsions on code by Roi A.K. 3/23-3/24 2021

This markdown is meant to keep track of questions and edits made to modules in the LocoA package.

*Questions and suggestions in the python files themselves were made in comments with brackets on top of this document*

*Code edits can be tracked in the pull request*

**This code is awesome! So much already done and working well**

## This document

Here I'll go systematically through the code in the LocoA package and for each class object and function within will track comments/questions/suggestions I had. 

In addition, I'll note edits to code I made.  

I hope this is helpful in organizing code and making it reproducible and accessible. If anything is unhelpful, please ignore me:)

---

## Locoa_alt.py

### Class simulator 

#### def __init__ 
     - would be good to have description of init here for reproducibility
     - line 49: when initializing arguments comment on each line to tell user (or future self) what each argument means
     - line 66: same as above  

#### def make_profile
     - line 75: description of function is very bare bones. What is the profile string? What is each parameter? What are the conditional statements doing? It's tedious but would be helpful to have more detail 

#### def format_data
     - would be good if description of overall function had clear stepwise workflow description: what goes in, what goes out, and how
       - For example, what are win_size and conv_size doing? 
     - line 102: question about `if` statement
     - line 112: what is the convolve function used for?  
     - line 118: what are your replacing here?
     - lines 119-120: what do 'h' and 'mNcell' stand for?
     - line 125: added night vs. day column (still not working)
     - lines 127-150: describe each chunk of code with the task it's doing


### Class Analysis

Will get to this soon...