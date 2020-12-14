# Title 1
Here you will find an overview of the methods used in the project. The data is generated using blablabla

## How to use
### 1. Generate the Galactic Center Excess Pseudo-data
Generate the GCE pseudo-data by running the generate.py module. The following parameters can be operated from the command line, if no arguments are given the script uses default values.

```
$ python generate.py -h

usage: generate.py [-h] [-n  number_of_maps] [-d  dimensions] [-dms dm_strength] [-mdm dm_mean] [-nl noise_level]
                   [-r  random_walk] [-s  shuffle_maps] [-p  PATH] [-v  version] [-vp variation_plane]
                   [-vgc variation_GC] [-gcs gc_scale]

Generate Galactic Center Excess pseudodata TBA

optional arguments:
  -h, --help           show this help message and exit
  -n  number_of_maps   The number of maps to generate (default: 1000)
  -d  dimensions       Dimensions of the maps use as: -d dim1,dim2,dim3 (default: 28,28,20)
  -dms dm_strength     Strength of dark matter (only relevant when using v1) (default: 1)
  -mdm dm_mean         Mean of dark matter spectrum (default: 0.0)
  -nl noise_level      Level of Gaussian noise in data (default: 0.008)
  -r  random_walk      Use random walk (default: True)
  -s  shuffle_maps     Shuffle the maps before saving (default: True)
  -p  PATH             Path to where the data should be stored (default: ../data/)
  -v  version          Choose the version of generator, v1:1 or v2:2 (default: 1)
  -vp variation_plane  Random variation in galactic plane normalization (propotion of the standard value)
                       (default: 0.0)
  -vgc variation_GC    Random variation in galactic center normalization (propotion of the standard value)
                       (default: 0.0)
  -gcs gc_scale        Scale of the galactic center (default: 2000000000000000.0)

```
if using `-v 1` the data is stored in a file called `data_(n,d1,d2,d3)_dm_nl_r_.npy` and when using `-v 2` the data is stored in `maps_(n,d1,d2,d3)_nl_r_.npy`.

### 2. Set up the configuration files and train
The two configuration files:
- Regression: config_regression.py
- Classification: config_classification.py

Holds the configurations for the regression and classification analysis that will be used for the CNN. To change dataset or some parts of the configuration such as number of epochs, layer configuration etc, edit these config scripts. Train the network by running the script CNN.py with flag c for classification and r for regression. If you want to save the model, use the n flag followed by the filename.

```
$ python CNN.py -h

usage: CNN.py [-h] [-r] [-c] [-n name] [-e [name]] [-v [V]]

This is the Convolutional neural network used in FYS-STK 4155
---------------------------------------------------------------------------
The configuration files that is used in this method are:

    - config_regression.py
    - config_classification.py.

In these files you can change datafile, epochs, loss function etc. Indicate
if you want to use the regression or classification option using the
corresponding flags. We recommend that you supply a filename such that the
CNN model is saved. This allows you to continue training at a later time,
and simplifies further analysis.

For more info: https://github.com/Alidafh/FYS-STK4155/tree/master/project3/code

optional arguments:
  -h, --help  show this help message and exit
  -n name     What name to store the model as (default: None)
  -e [name]   Continue training on given model (default: None)
  -v [V]      kFold validation, cannot be used with e (default: None)

required arguments:
  -r          Regression (default: False)
  -c          Classification (default: False)
```

Regression example: The command,

```
$ python CNN.py -rn reg2

```
will train a model that is saved as reg2. The path to where the model should be stored and the configuration of the network must be specified in config_regression.py. After training is completed, the model is stored as a folder called reg2, and the history(loss and metric as a function of epochs) is stored in a file called reg2_training.log.

If you are not satisfied with the results after the given number of epochs, you can resume training on the same model by doing: `python CNN.py -rn reg2 -e` or `python CNN.py -re reg2`.

If you are satisfied, you can perform 5-fold cross-validation with the command

```
$ python CNN.py -rn reg2 -v

```
You can choose a different number of folds by specifying an iteger after the validation flag. If you want 10 folds you would do `python CNN.py -rn reg2 -v 10`. Each model and training log of the folds are stored in the folder kFold_regression.


## Overview of files

| Files | Description |
| ------ | ------ |
| generate.py | script to create GCE pseudo-data |
| CNN.py | contains the convolutional neural network |
| config_regression.py |variables for the regression problem |
| config_classification.py |variables for the classification problem |
| regression.py | main script for regression |
| classification.py | main script for classsification |
