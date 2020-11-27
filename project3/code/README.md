# Title 1
Here you will find an overview of the methods used in the project. The data is generated using blablabla

## How to use
### 1. Generate the Galactic Center Excess Pseudo-data
Generate the GCE pseudo-data by running the generate.py module. The following parameters can be operated from the command line, if no arguments are given the script uses default values.

```
$ python generate.py -h
usage: generate.py [-h] [-n --number_of_maps] [-d --dimentions]
                   [-dm --dm_strength] [-nl --noise_level] [-r --random_walk]
                   [-s --shuffle_maps] [-p --PATH]

Generate Galactic Center Excess pseudodata TBA

optional arguments:
  -h, --help           show this help message and exit
  -n --number_of_maps  The number of maps to generate for each type,
                       default=1000
  -d --dimentions      Dimentions of the maps, default=(50,50,3)
  -dm --dm_strength    Strength of the dark matter, default=1
  -nl --noise_level    Level of gaussian nose in data, default=0.1
  -r --random_walk     Use random walk, default=True
  -s --shuffle_maps    Shuffle the maps before storing, default=True
  -p --PATH            Path to where the data should be stored,
                       default="../data/"
```

### 2. Run the analysis


## Overview of files

| Files | Description |
| ------ | ------ |
| generate.py | script to create GCE pseudo-data|

```
CODE
```
