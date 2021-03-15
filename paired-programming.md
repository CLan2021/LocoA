# Paired programming: ``LocoA``

### Goal

The objective of ``LocoA`` is clear from the proposal document: to analyze .txt files generated from the Trikinetics Locomotor Activity Monitor (LAM) device, as used on insect taxa.  The LAM data combines with CSV-formatted metadata to manipulate data into a format suitable for downstream plotting and statistical analyses.

The README.md document lists dependencies of ``LocoA`` and provides installation instructions.  Of note: the package ``os`` is part of the Python standard library, meaning that a user does not need to install it separately through conda.

### Data

The ``LocoA`` repository includes sample LAM datasets within a "data" folder, as well as the metadata CSV file and an additional README.md document that outlines the formatt of .txt files from the LAM system.  The repository also includes an example clustering plot and line plot.  An .ipynb file provides an overview of the code at present, including how these example plots are generated.

I appreciated the explanation of the LAM formatting given my limited familiarity with insect locomotion.  I imagine that most people seeking this package will be more familiar than I am, but just in case, the formatting info could be made more easily noticeable by moving it to the README.md document in the root of the repository.  In addition, while I was able to run the supplied sample datasets and metadata, the README.md could also provide apassage on how to generate these files and place them in the appropriate locations.

Currently, there is no description of what the example plots represent, except the ``LocoA`` aims to generate them from LAM data.  There is room to expound here.

### Code

The present code is completely functional: it can accept input data and metadata, manipulate it for sending to downstream analyses, and generate result plots.  We had discussed that the next step was to reorganize the code into classes and functions with the objective of improving readability.  I have created a separate script called ``locoa_alt.py`` to hold my changes, so that you can compare between the two.

### Contribution

I reorganized the code into two classes within the ``locoa_alt.py`` script.  The ``Simulator`` class object locates .txt files and processes them according to the metadata and user-supplied parameters.  The ``Analysis`` class object contains many more functions, all dedicated to producing plots or statistical analyses from the data.  ``Analysis`` is intended to use output from ``Simulator`` as input.  I have also included a Jupyter notebook called ``LocoA_reorganized`` to demonstrate how the reorganized code works.

This is just *one* way to reorganize the code; it's a setup that I found helpful in my thesis.  For example, you may find that the ``Analysis`` object could be further subdivided into more classes, with the functions distributed appropriately.  The code doesn't necessarily all need to be in one script, either; in my thesis I have a Simulator class and an Analysis class each in their own script.  The general benefit running through all the possibilities is that the code is easier to read, for users and for you as the developer.

### Suggestions

Some additional thoughts on improving the code from here:

1. I tried to add some comments to the best of my understanding of the analysis pipeline.  Adding comments for each "step" of the code's process helps to manage and understand the code's structure, and I recommend trying to make that a habit.
2. Some of the ``pandas`` code could probably be condensed with function chaining, like in the example from the spring break challenge notebook.
3. The variables ``group_idxs`` and ``group_idxs_`` (with an additional underscore) are necessarily distinct and used in different downstream functions.  Subjectively, I find this annoying to parse and might rename one or both of these variables, but I've seen this naming convention used elsewhere so it's up to you.
4. One of the stated goals of ``LocoA`` is to make the plotting functions more flexible.  In the reorganized code, the function level is where you could implement stylistic options.  For example: you could set up an optional argument for ``figsize`` which allows the user to input a desired plot size, instead of hard-coding a constant size into the function.
5. Double-check how/whether the functions ``make_profile`` and ``scatter3d``are intended to interact with other parts of the code.