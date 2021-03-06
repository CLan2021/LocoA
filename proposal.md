# LocoA - Locomotor Analysis

### What task/goal will the project accomplish and why is this useful?
When talking about analysis of locomotor activity rhythms, most works or analytical methods have been developed for *Drosophila* research. **LocoA** uses a combination of .txt output files from the Trikinetics Locomotor Activity Monitor (LAM) device and a meta dataset that includes information on each test subject to create plots, perform statistical analyses, and generate predictions. This program allows users to customize inputs and parameters when analyzing insect locomotor activity patterns.

### Where will the data come from? What type of data/input will a user provide to the program? 
The user will need to provide .txt output files generated automatically by the LAM  device and a meta dataset formatted in a tab-delimited CSV. For all the .txt activity files, be sure to align them such that each file begins at the same time of day. Examples of what the aligned .txt files and the meta dataset look like can be found in the data/ directory.

### How will a user interact with the program?
I am thinking of making **LocoA** into a CLI or API program where the user can edit the Python code directly to perform analyses or provide specific commands / arguments to the program.

### What type of output will the program produce (e.g., text, plots)?
**LocoA** will produce various plots (scatter plots and line graphs) and results of statistical analyses as well as model predictions. 

<img src="scatterplot.png" width="100" height="100">

<img src="lineplot.png" width="100" height="100">

### What other tools currently exist to do this task, or something similar?
There is a program called [`ShinyR-DAM`](https://github.com/KarolCichewicz/ShinyR-DAM.git) that operates in the cloud or can be downloaded and run locally using RStudio. While this program also produces plots and summary tables, it is only designed to analyze data recorded by the Drosophila Activity Monitor (DAM) system (Trikinetics, Waltham, MA) and has limited or fixed options for additional parameters.


**LocoA** fills a niche that other existing programs do not by allowing the user to customize input parameters and edit the scripts in Python.
