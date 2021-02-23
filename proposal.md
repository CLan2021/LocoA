# package for locomotor activity analysis
An outline of a new project idea (for now)


# Comments
## Questions
- What other methods currently exist to analyze this type of data? How will your program fill a niche that these other program do not? Example, it will have more options? or it will be easier to use? Or it will be in Python?

- What are some examples of circadian rhythm or 3D scatter analysis plots you wish to produce? What type of statistical inference tools in scikit-learn are best suited for this type of analysis? Perhaps look into clustering tools for the scatter plots, and gaussian process analyses for the circadian rhythms?


## Generative data
I don't know much about the LAM data type. Can you provide links to some example data files? If possible, it would be great to be able to simulate data under a generative model to show that your analyses are working properly. For example, simulate data with known values for parameter X, Y, Z, and then fit your models and show that they accurately infer X,Y,Z.


## Class objects
How will you structure your code? Perhaps (1) a class for loading LAM and CSV data files and checking they are formatted appropriately; (2) a class for performing circadian rhythm analyses, generating these types of plots, and statistics; and (3) a class for 3D scatterplot analyses, with options to modify style settings for the output plots.



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


