# Regression analysis and resampling methods

## Overview of files:
Files in this routine\
plotting.py: Contains the functions for creating figures\
general_functions.py: Contains the analysis functions\
helpers.py: Supergeneral functions such as SVD etc.\
main.py: the main script\
run.sh: Simple bash script to run main.py\
clean.sh: Bash script which deletes the output folders etc. in case a clean run is wanted.\    

## How to run:
Run the analysis using

```
source run.sh
```

If they do not exist already, the following folders are created:
- output/figures

NOTE: If you want to run the whole shabang, uncomment line 42 and 77 in main.py

## Credits
Builds on example-code used/created by Morten Hjorth-Jensen for the class FYS-STK4155.

### TO-DO:
a) Ordinary least squares on the Franke function
  - [x] Generate data for the franke function (can this be done differently?)
  - [x] Function to generate Design matrix
  - [x] Split and Scale data
  - [x] Function for OLS
  - [x] MSE, R2-score, bias and variance
  - [ ] Confidence interval for the regression parameters

b) Bias-variance trade-off and resampling techniques\
  - [ ] Bootstrap
c) Cross-validation as resampling techniques\
d) Ridge regression on the Franke function with resampling\
e) Lasso Regression on the Franke function with resampling\
f) Introducing real data and preparing the data analysis\
g) OLS, Ridge and Lasso regression with resampling\


### Work-distribution

### Notes
Alida: as long as n>p the two OLS functions returns the same values (both calculate the pseudoinverse I guess). Having trouble with the
Alida: I started working on the bootstrapping
