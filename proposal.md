# package for locomotor activity analysis
An outline of a new project idea (for now)


# Comments
- What task/goal will the project accomplish and why is this useful?
(this should be a concise statement that would convince a user that your program is worth their time to read more about and try using.


- What type of data/input will a user provide to the program?


- Where will the data come from?


- How will a user interact with the program?


- What type of output will the program produce (e.g., text, plots)?


- What other tools currently exist to do this task, or something similar?


- What other methods currently exist to analyze this type of data? How will your program fill a niche that these other program do not? Example, it will have more options? or it will be easier to use? Or it will be in Python?

- What are some examples of circadian rhythm or 3D scatter analysis plots you wish to produce? What type of statistical inference tools in scikit-learn are best suited for this type of analysis? Perhaps look into clustering tools for the scatter plots, and gaussian process analyses for the circadian rhythms?


## Generative data
I don't know much about the LAM data type. Can you provide links to some example data files? If possible, it would be great to be able to simulate data under a generative model to show that your analyses are working properly. For example, simulate data with known values for parameter X, Y, Z, and then fit your models and show that they accurately infer X,Y,Z.


## Class objects
How will you structure your code? Perhaps (1) a class for loading LAM and CSV data files and checking they are formatted appropriately; (2) a class for performing circadian rhythm analyses, generating these types of plots, and statistics; and (3) a class for 3D scatterplot analyses, with options to modify style settings for the output plots.


# Proposal
- What task/goal will the project accomplish and why is this useful?

The project will provide tools to analyze Locomotor Activity Monitor (LAM) data and return plots of daily circadian rhythm, 3D scatter analysis, etc. of activity counts. It is especially useful when the datasets start to get big, being able to automatize the process for cleaning, parsing, and analyzing data.

- What type of data/input will a user provide to the program?

The types of data/input are .txt files and a meta dataset organized in a CSV file.

- Where will the data come from?

The data will come from user's input (e.g. CSV file, cleaned .txt output file from the LAM device)

- What type of output will the program produce (e.g., text, plots)?

The program will produce 3D scatter plots, multi line plots, ....?

- What other tools currently exist to do this task, or something similar?

There's a program called ShinyR-DAM that uses .txt output files from the *Drosophila* Activity Monitor (DAM) system, producing customizable plots and CSV files.

### Description of project goal
My project will provide tools to analyze Locomotor Activity Monitor (LAM) data and return plots of daily circadian rhythm, 3D scatter analysis, etc. of activity counts.


### Description of the code:
I will be using the following packages to organize data sets and perform statistical analyses as well as generate plots.
`os`: to perform operating system tasks.
`pandas`: to organize and analyze data.
`numpy`: to organize and analyze data.
`scikit-learn`: to perform statistical analyses and train and predict models.
`pyplot`: to create graphs.


### Description of the data:
I will be using standard .txt files generated from the Trikinetics Locomotor Activity Monitor device along with CSV data input by the user.


### Description of user interaction:
The user will need to organize and align .txt files, whereas meta-dataset (CSV file) with data labels and parameters can vary according to the user's needs.


