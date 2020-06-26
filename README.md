### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Project Motivation <a name="motivation"></a>

This is ... project. I was interestested in using ... data to better understand:

1. Q1
2. Q2
3. Q3

## File Descriptions <a name="files"></a>

There is one notebook available here to showcase work related to the above questions. The notebook is exploratory in searching through the data pertaining to the questions showcased by the notebook title.  Markdown cells were used to assist in walking through the thought process for individual steps.  

There are ... data files used in the notebook. "..." is the data file for... .

The data file is available at [website](http://....).

## Results <a name="results"></a>

The main findings of the code is ....

After additional exploring I think that we can safely change value 2 of "related" column to 0, because rows with values 0 and 2 have the same pattern of class label values - if "related" column has one of these values, then all another class labels are equal 0 (which is not true for value 1).

Also this finding means that we can predict at first whether the message is related (related=1), and then predict all other class labels, otherwise (related=0) we can zero all another class labels.

## Licensing, Authors, Acknowledgements <a name="licensing"></a>

Must give credit to .... for the data.  You can find the Licensing for the data and other descriptive information at the link available [here](http://...).  Otherwise, feel free to use the code here as you would like! 
