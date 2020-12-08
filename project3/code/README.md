# Title 1
Here you will find an overview of the methods used in the project. The data is generated using blablabla

## How to use
### 1. Generate the Galactic Center Excess Pseudo-data
Generate the GCE pseudo-data by running the generate.py module. The following parameters can be operated from the command line, if no arguments are given the script uses default values.

```
$ python generate.py -h
usage: generate.py [-h] [-n --number_of_maps] [-d --dimensions]
                   [-dm --dm_strength] [-nl --noise_level] [-r --random_walk]
                   [-s --shuffle_maps] [-p --PATH]

Generate Galactic Center Excess pseudodata TBA

optional arguments:
  -h, --help           show this help message and exit
  -n --number_of_maps  The number of maps to generated for each type 
  			(so total number of maps is 2n),
                       default=1000
  -d --dimensions      Dimensions of the maps, default=(28,28,10)
  -dm --dm_strength    Strength of the dark matter, default=1
  -nl --noise_level    Level of gaussian noise in data, default=0.1
  -r --random_walk     Use random walk, default=True
  -s --shuffle_maps    Shuffle the maps before storing, default=True
  -p --PATH            Path to where the data should be stored,
                       default="../data/"
```
The data is stored in a file called `data_(n,d1,d2,d3)_dm_nl_r_.npy`.

### 2. Run the analysis

The analysis consists of the three scripts:

- CNN.py
- configuration.py
- main.py

run main.py. To change dataset or some parts of the configuration of the CNN such as number of epochs, layer configuration etc, edit the configuration.py script.

## Overview of files

| Files | Description |
| ------ | ------ |
| generate.py | script to create GCE pseudo-data |
| CNN.py | contains the convolutional neural network |
| configuration.py | contains the variables for the neural network |
| main.py | main script |

```
CODE
```
# TO DO

- Maps too similar causes overtraining?
- python generate.py -n 5000 -d 28,28,10
