# Regression analysis and resampling methods

## Overview of files:

plotting.py: Contains the graphic functions\
functions.py: Contains the analysis functions\
tools.py: general tools such as SVD etc.\
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
- output/outfiles

## Credits
Builds on example-code used/created by Morten Hjorth-Jensen for the class FYS-STK4155.

### TO-DO:
a) Ordinary least squares on the Franke function
  - [x] Generate data for the franke function
  - [x] Function to generate Design matrix
  - [x] Split and Scale data
  - [x] Function for OLS
  - [x] MSE, R2-score, bias and variance
  - [ ] Confidence interval for the regression parameters

b) Bias-variance trade-off and resampling techniques\
  - [x] Bootstrap implementation (done I think)
  - [x] Bias-Variance plotting
  - [ ] Comparisons datasize

c) Cross-validation as resampling techniques\
  - [ ] kFold stuff
  - [ ] Cross-validate

d) Ridge regression on the Franke function with resampling\
  - [ ] Function that does Ridge regression

e) Lasso Regression on the Franke function with resampling\
  - [ ] Function that does Lasso regression

f) Introducing real data and preparing the data analysis\
g) OLS, Ridge and Lasso regression with resampling\


### Work-distribution

### Notes
As long as n>p the two OLS functions returns the same values (both calculate the pseudoinverse I guess). Creates trouble with the confidence intervals when p>n, temporarily bypassing by taking the absolute value, but should find some more stable solution? \
