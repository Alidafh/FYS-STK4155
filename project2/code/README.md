# Overview of files
Here are an overview of files in this routine, the Neural Network and regression classes are contained in:

| Files | Description |
| ------ | ------ |
| tools.py | Contains various helper functions|
| skynet | Contains the neural network and regression classes |
| mnist_loader | needed to load the mnist data, written by M. Nielsen |

The following files are examples of use and are what we have used to generate the results for the report in the report folder:

| Files | Description |
| ------ | ------ |
| QuickPlot  | Classes made for convenient plotting |
| super_main | Main script for making regression results |
| mega_main | main script for making classification results |
| gradient.py | Script to generate results using SGD with different parameter choices specified in the command line.|
| run_gradient.py | run the gradient script to generate the results we need for plotting |
| stochastic.py | plotting the results |

## Use: gradient.py
Script to generate results using SGD with different parameter choices specified in the command line. Path to where results should be stored must be specified on line 119.

```
$ python gradient.py -h
usage: gradient.py [-h] [-r --method] [-l --lambda] [-d --degree]
                   [-ep --n_epochs] [-bs --batch_size] [-lr --learn_rate]
                   [-gm --gamma] [-p --print]

Use Stochastic Gradient Descent to find the beta values either for OLS or
Ridge loss functions. A log file of the loss for each epoch number and batch
number is created and stored in output/SGDLOG_*.txt

optional arguments:
  -h, --help        show this help message and exit
  -r --method       The regression method, options are [OLS] and [Ridge],
                    default=OLS
  -l --lambda       The lambda value for ridge regression, default=0.001
  -d --degree       Polynomial degree of design matrix, default=4
  -ep --n_epochs    The number of epochs, default=100
  -bs --batch_size  Size of the minibatches, default=5
  -lr --learn_rate  The learning rate, default=None
  -gm --gamma       The gamma value for momentum, default=None
  -p --print        If the loss should be printed, default=True
```
An example of use with polynomial degree 7, learning rate 0.1 and no printing of the loss for each epoch:

```
$ python gradient.py -d 7 -lr 0.1 -p False
Method:      OLS
Lambda:      0.001 (Only if Ridge)
Polydegree:  7
Epochs:      100
Batch size:  5
Learn rate:  0.1
Gamma:       None


Stochastic Gradient Descent
-----------------------------------
Beta :
[ 0.90698779 -0.13168087  0.16240147 -0.60593585  0.24466069 -1.0313762
  0.7159044   0.61156063  0.32509254  0.61123802 -0.30067928 -0.29530937
  0.53717894 -0.84048048 -1.20301778  0.00923838 -0.62196817  0.58032093
 -0.50070325 -0.36609728  1.98845799  0.3285099   0.48929266 -1.18849564
 -0.38483564  0.38950268 -0.34724724  0.63244565 -0.29289879  0.04160347
 -0.44519039  1.85353318 -0.08500841 -1.04086528  1.27995735 -1.35021494]
r2 :  0.8145284504241741
mse:  0.01817967105506659


Ordinary Least Squares:
-----------------------------------
Beta :
[   0.7046924    -1.08443371   -0.23423453   15.53243677   11.98080575
    3.98576943  -76.57260961  -57.76153637  -37.18868615  -20.28585757
  166.05138506  126.42347823  125.28562642   53.6201671    36.05430369
 -177.70797094 -156.51247306 -147.69402204 -123.317931    -51.83815494
  -22.98744591   90.84065202  109.65090088   64.00941418  105.89535281
   50.59318199   34.62478502   -0.53311264  -17.24219902  -34.01678051
   -2.68411865  -33.61985023  -16.80285572  -10.6602168    -9.87056785
    3.85305809]
r2 :  0.8958937749159206
mse:  0.010204351724788552

#############################################################

```

## Credit
This code builds on example-code used/created by Morten Hjorth-Jensen for the class [FYS-STK4155](https://github.com/CompPhysics/MachineLearning/)


It also uses portions of the 'mnist_loader.py' script written by Michael Nielsen:


MIT License

Copyright (c) 2012-2018 Michael Nielsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
