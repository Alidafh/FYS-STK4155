# Regression analysis and resampling methods

## Overview of files:

- plotting.py: Contains the graphic functions
- functions.py: Contains the analysis functions
- tools.py: general tools such as SVD etc.
- main.py: the main script
- run.sh: Simple bash script to run main.py
- clean.sh: Bash script which deletes the output folders etc. in case a clean run is wanted.

## How to run:
Run the analysis with the command

```
$ source run.sh
```

If they do not exist already, the following folders are created:
- output/figures
- output/outfiles

If they do exist, this script will save figures and data in the existing folders.

The script clean.sh removes the folders created by run.sh and can be executed by writing:

```
$ source clean.sh
```

**NOTE**: clean.sh deletes the output folders. If you already have a personal folder named output which is unrelated to this project, you should rename it.

## Credit

Builds on example-code used/created by Morten Hjorth-Jensen for the class FYS-STK4155.

### TO-DO:
a) Ordinary least squares on the Franke function
  - [x] Generate data for the franke function
  - [x] Function to generate Design matrix
  - [x] Split and Scale data
  - [x] Function for OLS
  - [x] MSE, R2-score, bias and variance
  - [x] Confidence interval for the regression parameters

b) Bias-variance trade-off and resampling techniques\
  - [x] Bootstrap implementation (done I think)
  - [x] Bias-Variance plotting
  - [ ] Comparisons data-size

c) Cross-validation as resampling techniques\
  - [x] kFold
  - [ ] Cross-validate

d) Ridge regression on the Franke function with resampling\
  - [ ] Function that does Ridge regression

e) Lasso Regression on the Franke function with resampling\
  - [ ] DO lasso stuff

f) Introducing real data and preparing the data analysis\
g) OLS, Ridge and Lasso regression with resampling\


### Notes
- functions.py -> metrics(): when using this function to calcualte the R2 score with the bootstrap resampling, it is waaay off
- The bias and MSE caclulated using kFold (n=100 datapoint, k=5 folds) are completely identical, both when using our formulas and sklearn.. why?  
