## README

### Team

This repository contains the code for the project1 of the Machine Learning Course. The team is composed of

   - Benedek Harsányi (xxxxxx)
   - kamil czerniak (xxxxxx)
   - Mohamed Allouch (342708)

### Dependencies

The project was written with Python 3.8, and the only external library used to generate the predictions is `numpy`. To install it run

    - pip: pip install numpy
    - Anaconda: conda install numpy

NOTE: For the generation of the plots it is also necessary to use the package `matplotlib`. While this is not necessary to generate the file `OUTPUT.csv`, to be able to generate the plots it must be installed

    - pip: pip install matplotlib
    - Anaconda: conda install matplotlib


### Reproducing Results

In order to reproduce the results obtain, one can get the original untreated data at https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files.
This data should be placed inside a folder called `data` on the same root as the file `run.py`. Then, to reproduce the .csv file generated,
it is simply a matter of running

`python run.py`

on the terminal or powershell. This will generate a file called `OUTPUT.csv` with exactly the same results as presented on AIcrowd.

NOTE: When running the script it may give a deprecation warning about ragged nested sequences. However, the program runs regardless and it
does not affect the final output.


### Repo Architecture

   - run.py: Final script used to replicate the predictions submitted, written on the file OUTPUT.csv
   - OUTPUT.csv: File with the predictions generated from the test data

<pre>  
├─── data
    ├─── test.csv : contains the test data we want to use to generate the predictions
    ├─── train.csv : contains the data used to train the model
├─── datacleanup.py : auxiliary functions used for data preprocessing(removing outliers, data standarisation, data spliting ...)
├─── helpers.py : auxiliary functions, like  load data, make submissions ...
├─── implementations.py : all the implementations required for the project
├─── OUTPUT.csv : predictions generated from the test data
├─── README.md : README
├─── run.py : final script used to replicate the predictions submitted, written on the file OUTPUT.csv
└─── visualizations.py : functions to produce the plots for visualization of relevant parameters (correlation heatmap, feature distribution ... )
</pre>


### Technical Details

##### Data Cleaning Procedure

The training dataset provided had to be cleaned up in several fronts.
