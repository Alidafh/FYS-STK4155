# FYS-STK4155 - Project 3: Studying the potential of convolutional neural networks in identifying a dark matter signal in gamma rays from the galactic center

In this project we investigate the possibility of using a convolutional neural network to find an excess of $\gamma$-ray emission, possibly originating from dark matter, in the Galactic Center. The work is in large part inspired by the article by an article by \citet{GCE_deep_learning} which contains a proof of concept of using convolutional neural networks to investigate the origin of an excess emission of $\gamma$ rays in the direction of the Galactic Center. We will first treat this as a classification problem, where we train a network to classify wether a galaxy is created with or without a dark-matter excess, before we move on to a regression based problem where we try to train a model to predict the strength of an inserted signal. Our method shows that it is possible to use a convolutional neural network to classify galaxies with an excess, where we achieve an accuracy of $99.85\%$ and a binary cross-entropy loss of $0.0046$ when classifying data that has the maximum realistic signal strength $f_{dms}=1$, and an accuracy of $97.90\%$ when using $f_{dms}=0.5$. When using the convolutional neural network to predict the $f_{dms}$ strength of a galaxy, our best model has an $R^2$-score of $0.88$ and mean squared error loss of $0.0091$.  

## Folders

- code: Contains all the code files, the minst data file, a full description of the code can be found in a separate README file in this folder.
    - fits: Contains fits files that are used in the program to generate simulations of the Galactic Center
- data: Contains the data files
- report: Contains the report file
- figures: Contains the figures generated for our report


## Development
