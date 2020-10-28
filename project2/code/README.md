# Overview of files

| File | Description |
| ------ | ------ |
| regression.py | Contains the Regression class |
| tools.py | Contains various helper functions|
| gradient.py | Preform SGD |
| main.py| Reads the logfile and plots the loss (temporary) |

## Use
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


## To-do
- [ ] finish plotting for SGD
