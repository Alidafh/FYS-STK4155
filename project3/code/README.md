# Title 1
Here you will find an overview of the methods used in the project. The data is generated using blablabla

## How to use
### 1. Generate the Galactic Center Excess Pseudo-data
Generate the GCE pseudo-data by running the generate.py module. The following parameters can be operated from the command line, if no arguments are given the script uses default values.

```
$ python generate.py -h

usage: generate.py [-h] [-n number_of_maps] [-d dimentions] [-dm dm_strength]
                   [-nl noise_level] [-r random_walk] [-s shuffle_maps]
                   [-p PATH] [-v version]

Generate Galactic Center Excess pseudodata TBA

optional arguments:
  -h, --help          show this help message and exit
  -n  number_of_maps  The number of maps to generate (default: 1000)
  -d  dimentions      Dimentions of the maps use as: -d dim1,dim2,dim3 (default: 28,28,10)
  -dm dm_strength     Strength of dark matter (only relevant when using v1) (default: 1)
  -nl noise_level     Level of gaussian nose in data (default: 1)
  -r  random_walk     Use random walk (default: True)
  -s  shuffle_maps    Shuffle the maps before saving (default: True)
  -p  PATH            Path to where the data should be stored (default: ../data/)
  -v  version         Choose the version of generator, v1:1 or v2:2 (default: 1)
```
if using `-v 1` the data is stored in a file called `data_(n,d1,d2,d3)_dm_nl_r_.npy` and when using `-v 2` the data is stored in `maps_(n,d1,d2,d3)_nl_r_.npy`.

### 2. Set up the configuration files and train
The two configuration files:
- Regression: config_regression.py
- Classification: config_classification.py

Holds the configurations for the regression and classification analysis that will be used for the CNN. To change dataset or some parts of the configuration such as number of epochs, layer configuration etc, edit these config scripts. Train the network by running the script CNN.py with flag c for classification and r for regression. If you want to save the model, use the n flag followed by the filename.

```
$ python CNN.py -h

usage: CNN.py [-h] [-r] [-c] [-n name]

Train the CNN

optional arguments:
  -h, --help  show this help message and exit
  -n name     What name to store the model as (default: None)

required arguments:
  -r          Regression (default: False)
  -c          Classification (default: False)
```

Example: to use classification do,
```
python CNN.py -cn filename
```

## Overview of files

| Files | Description |
| ------ | ------ |
| generate.py | script to create GCE pseudo-data |
| CNN.py | contains the convolutional neural network |
| config_regression.py |variables for the regression problem |
| config_classification.py |variables for the classification problem |
| main.py | main script |

```
CODE
```
# TO DO

- Maps too similar causes overtraining?
- python generate.py -n 5000 -d 28,28,10
