# Overview of files

| File | Description |
| ------ | ------ |
| tools.py | Contains various helper functions|
| gradient.py | Preform SGD |
| QuickPlot  | Classes made for convenient plotting |
| super_main | Main script for making regression results |
| skynet | Contains the neural network and regression classes |
| mega_main | main script for making classification results |
| mnist_loader | needed to load the mnist data, written by M. Nielsen |
| ------ | ------ |
| sgd_v1/regression.py | The old regression class |
| sgd_v1/gradient.py | same as the other gradient.py |
| sgd_v1/run_gradient.py | run the gradient script to generate the results we need for plotting |
| sgd_v1/stochastic.py | plotting the results |

## Example of use: gradient.py
```
$ python gradient.py -h
usage: gradient.py [-h] [-r --method] [-l --lambda] [-d --gamma]
                   [-ep --n_epochs] [-bs --batch_size] [-lr --learn_rate]
                   [-gm --gamma]

Use Stochastic Gradient Descent to find the beta values either for OLS or
Ridge loss functions. A log file of the loss for each epoch number and batch
number is created and stored in output/data/SGDLOG_*.txt

optional arguments:
  -h, --help        show this help message and exit
  -r --method       The regression method, options are [OLS] and [Ridge]
  -l --lambda       The lambda value for ridge regression
  -d --gamma        Polynomial degree of design matrix
  -ep --n_epochs    The number of epochs
  -bs --batch_size  Size of the minibatches
  -lr --learn_rate  The learning rate
  -gm --gamma       The gamma value for momentum
```

## Credit
This code builds on example-code used/created by Morten Hjorth-Jensen for the class [FYS-STK4155](https://github.com/CompPhysics/MachineLearning/)


It also uses portions of the 'mnist_loader.py' script written by Michael Nielsen:\


MIT License\

Copyright (c) 2012-2018 Michael Nielsen\

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
