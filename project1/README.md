# Regression analysis and resampling methods

## Overview of files:
Files in this routine\
plotting.py: Contains the functions used to create figures\
general_functions.py: Contains the analysis functions\
main.py: the main script
run.sh: Simple bash script to run main.py
clean.sh: Bash script which deletes the output folders etc. in case a clean run is wanted.    

## How to run:
Run the analysis using

```
source run.sh
```

If they do not exist already, the following folders are created:
- output/figures

## Credits
If code is sourced from someone else

### TO-DO:
a) Ordinary least squares on the Franke function
  - [x] Generate data for the franke function
  - [x] Function to generate Design matrix
  - [x] Split data into training set and a test set
  - [ ] Scale data
  - [x] Function for OLS
  - [x] MSE, R2-score and variance

b) Bias-variance trade-off and resampling techniques\
c) Cross-validation as resampling techniques\
d) Ridge regression on the Franke function with resampling\
e) Lasso Regression on the Franke function with resampling\
f) Introducing real data and preparing the data analysis\
g) OLS, Ridge and Lasso regression with resampling\


### Workdistribution

### Notes
