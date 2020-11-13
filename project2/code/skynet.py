#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:23:27 2020

@author: Gert Kluge and Alida Hardersen
"""

import numpy as np
import tools as tools
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from numba import jit
import numba as nb

class Regression:
    def __init__(self):
        self.beta = None            # Regression coefficients
        self.beta_var=None          # Variance of the regression coefficients
        self.residual_sum_squares = None
        self.residuals = None
        self.ndf = None
        self.loss = None
        self.loss_SGD = None
        self.regularization_type = L2
        self.regularize_indices = []
        self.accuracy = []

    def regularize_index(self,index):
        if self.regularize_indices == []:
            return 1
        else:
            return self.regularize_indices[index]

    def predict(self, X):
        """Get the predicted value"""
        y_predict = X @ self.beta
        return y_predict

    def residual(self, X, y):
        """ Get the residuals, RSS, effective ndf and sigma_hat"""
        y_predict = self.predict(X)
        self.residuals = y_predict - y
        self.residual_sum_squares = sum(self.residuals**2)
        self.ndf = len(y) - len(self.beta)
        self.loss = self.residual_sum_squares/self.ndf
        self.sigma_hat = self.residual_sum_squares/self.ndf

    def mse(self, X, y):
        """ The Mean squared error """
        y_predict = self.predict(X)
        mse = np.mean(np.mean((y - y_predict)**2))
        return mse

    def r2score(self, X, y):
        """ The explained R2 score """
        y_predict = self.predict(X)
        r2score = 1 - ((np.sum((y - y_predict)**2))/(np.sum((y - np.mean(y))**2)))
        return r2score

    def Bootstrap(self, X, y, nbs, ts=0.2):
        """
        Bootstrap resampling method

        Input:
        --------
        X: ndarray, shape (n,p)
            Full design matrix

        y: ndarray, shape (n,)
            Full array of the response variable

        nbs: int
            The number of bootstrap loops

        ts: float, default=0.2
            The desired size of the test-set

        Returns:
        ---------
        r2: ndarray, shape (nbs,)
            Explained R2-score as a function of number of bootstraps

        mse: ndarray, shape (nbs,)
            Mean squared error as a function of number of bootstraps

        var: ndarray, shape (nbs,)
            Variance as a function of number of bootstraps

        bias: ndarray, shape (nbs,)
            Bias as a function of number of bootstraps
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts)
        X_train, X_test = tools.scale_X(X_train, X_test)

        r2_ = np.zeros(nbs)
        mse_ = np.zeros(nbs)
        var_ = np.zeros(nbs)
        bias_ = np.zeros(nbs)

        for i in range(nbs):
            X_, y_ = resample(X_train, y_train)
            self.fit(X_, y_)
            y_predict = self.predict(X_test)
            r2_[i] = self.r2score(X_test, y_test)
            mse_[i] = self.mse(X_test, y_test)
            var_[i] = np.mean(np.var(y_predict))
            bias_[i]= np.mean((y_test - np.mean(y_predict))**2)

        r2 = np.mean(r2_)
        mse = np.mean(mse_)
        var = np.mean(var_)
        bias = np.mean(bias_)
        return r2, mse, var, bias

    @nb.jit(forceobj=True)
    def SGD(self, X, y, n_epochs=50, batch_size=5, learn_rate=None, gamma=None, prin=True,
            accuracy=False, test_data=None, lamb = 0, stop_threshhold=5, stop_criterion="mse"):
        """
        Fit the model by minimization using Stochastic Gradient Descent.

        Parameters
        ----------
        X : ndarray, shape (n,p)
            Design matrix that contains the input features

        y : ndarray, shape (n,)
            The response datapoints

        n_epochs : int, default=50
            The number of iterations over each minibatch

        batch_size : int, default=5
            The size of each minibatch

        learn_rate : float, default=None
            The learning rate, if no learning rate is specified a learning
            schedule is used based on the number of epochs and the batch size.

        gamma : float, default=None
            The gamma value for a momentum based SGD, if no gamma value is
            specified, a non-momentum based SGD is used.

        prin : bool, default=False
            If set to True a progress report is printed of the loss for each
            epoch and batch number.

        accuracy: bool, default=False
            TBD

        test_data: ndarray, shape (n,), default=None
            Test data can be supplied as a means for validation

        lamb: float, default=0
            Regularization parameter for l1 or l2 regularization

        stop_threshhold: int, defualt=5
            TBD

        stop criterion: string, default="mse"
            The stopping criterion used, TBD


        Attributes
        ----------
        beta : ndarray, shape (p,)
            Array of regression parameters

        loss_: ndarray, shape (n_epoch, )
            Training loss as a function of epochs

        mse_: ndarray, shape(n_epoch, )
            Training MSE as a function of epochs

        R2_: ndarray, shape (n_epoch, )
            Explained R2-score as a function of epochs

        loss_test: ndarray, shape(n_epoch, )
            Validation/test loss as a function of epochs

        mse_test: ndarray, shape(n_epoch, )
            Validation/test MSE as a function of epochs

        R2_test: ndarray, shape(n_epoch, )
            Validation/test explained R2-score as a function of epochs

        """

        n = len(X)              # number of datapoints
        p = len(X[0])           # number of parameters
        m = int(n/batch_size)   # number of minibatches

        np.random.seed(42)
        if self.beta == None: self.beta = np.random.randn(p)
        momentum = list(np.zeros(len(self.beta)))
        step_size = list(np.zeros(len(self.beta)))
        loss_ = list(np.zeros(n_epochs))
        mse_ = list(np.zeros(n_epochs))
        R2_ = list(np.zeros(n_epochs))

        loss_test = list(np.zeros(n_epochs))
        mse_test = list(np.zeros(n_epochs))
        R2_test = list(np.zeros(n_epochs))

        stop_cri = -np.array(self.accuracy) if stop_criterion=="accuray" else mse_
        ep = 0

        while not check_if_stop(ep, n_epochs, mse=stop_cri, threshhold=stop_threshhold):
            X_shuffled = list(np.zeros(len(X)))
            y_shuffled = list(np.zeros(len(y)))

            data = list(zip(X,y))
            np.random.shuffle(data)

            for i, (Xs, ys) in enumerate(data):
                X_shuffled[i] = Xs
                y_shuffled[i] = ys

            X_batches = [X_shuffled[k:k+batch_size] for k in range(0, len(X), batch_size)]
            y_batches = [y_shuffled[k:k+batch_size] for k in range(0, len(X), batch_size)]

            for i, (x_batch, y_batch) in enumerate(list(zip(X_batches,y_batches))):
                gradient = self.gradient(x_batch, y_batch)
                len_batch = len(x_batch)
                beta_temp = self.beta
                lr = learn_rate if learn_rate else tools.learning_schedule(ep*m+i, 5, 50)

                for k, el in enumerate(beta_temp):
                    momentum[k] = gamma*momentum[k] + (1 - gamma)*gradient[k] if gamma else gradient[k]
                    step_size[k] = momentum[k]*lr
                    beta_temp[k] = el - step_size[k]/len_batch - self.regularize_index(k)*self.regularization_type(el)*lamb*lr/n

                self.beta = beta_temp

            self.residual(X,y)
            loss_[ep] = np.sum(self.loss)
            mse_[ep] = self.mse(X,y)
            R2_[ep] = self.r2score(X,y)

            info = "Epoch: {:}/{:}\t\t Loss: {:.5f}".format(ep+1, n_epochs, loss_[ep])

            if accuracy:
                y_new=np.zeros(len(y))
                for u, elem in enumerate(y):
                    for v, dummy in enumerate(elem):
                        if elem[v] == 1:
                            y_new[u] = v

                test_data_temp_train = list(zip(X,y_new))
                current_accuracy_train = self.evaluate(test_data_temp_train)
                self.accuracies_train.append(current_accuracy_train/len(test_data_temp_train))

            if test_data:

                xt, yt, test_data = split_data(test_data, keep=True)

                if accuracy:
                    yr = [np.zeros(shape=(10,1)) for yi in yt]
                    for i, yi in enumerate(yt):
                        yr[i][yi] = 1
                    yr = np.array(yr)
                else:
                    yr = yt

                self.residual(xt,yr)
                loss_test[ep] = np.sum(self.loss)
                mse_test[ep] = self.mse(xt,yr)
                R2_test[ep] = self.r2score(xt,yr)

                info = info + "\t\t Test Loss: {:.5f}".format(loss_test[ep])

                if accuracy:

                    test_data_temp = list(zip(xt,yt))
                    current_accuracy = self.evaluate(test_data_temp)
                    self.accuracies.append(current_accuracy/len(yr))
                    more_info = "\t\t Accuracy: {:} / {:}".format(current_accuracy, len(yr))
                    info = info + more_info


            if prin: print(info)
            ep += 1

        print()

        self.loss_ = loss_[:ep-1]
        self.mse_ = mse_[:ep-1]
        self.R2_ = R2_[:ep-1]

        self.loss_test = loss_test[:ep-1]
        self.mse_test = mse_test[:ep-1]
        self.R2_test = R2_test[:ep-1]

        return loss_


class Network(Regression):
    """
    Class containing the Neural Network.
    The Network uses a SGD method for training, and can be initialized with
    various activation functions and used both for classification and
    for linear regression. The default activation function is sigmoid.

    Parameters:
    ----------
    layer_sizes: list [input, hidden nodes, output]

    """

    def __init__(self, layer_sizes):

        self.n_layers = len(layer_sizes)

        self.layer_sizes = layer_sizes

        # First layer is input, thus no biases.
        self.biases = [np.random.randn(y,1) for y in layer_sizes[1:]]

        # Vector of matrices, where columns are the node within the erlier layer,
        # rows the node in the affected layer, and values are weights.
        self.weights = [np.random.randn(y,x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

        self.len_wb = self.find_len_wb()

        self.beta = self.wb_to_beta()

        self.standard_af = Sigmoid()

        self.afs = None

        self.set_afs_to_standard()

        self.regularization_type = L2

        self.regularize_indices = [i<len(self.weights) for i in range(self.len_wb)]

        self.accuracies = []

        self.accuracies_train = []



    def predict(self, x):
        y = list(np.zeros(len(x)))

        for i,el in enumerate(x):
            y[i] = self.feedforward(el)

        return np.array(y)


    def feedforward(self, a):
        # input should be a numpy ndarray
        s = -1
        for b, w in zip(self.biases, self.weights):
            s += 1
            a = self.afs[s].func(np.dot(w, a) + b)

        return a


    def train(self, x, z, n_epochs=30, batch_size=10, learn_rate=3, gamma=None, prin=True, accuracy=False, test_data=None,
              lamb = 0,stop_threshhold=5, stop_criterion="mse"):
        """
        The function that trains the neural network. In practice takes the
        parameters that have been optimized by the SGD function and uses them
        to update the network. Input parameters are the same as for Regression:
        SGD method.
        """

        self.SGD(x,z, n_epochs=n_epochs, batch_size=batch_size, learn_rate=learn_rate, gamma=gamma, prin=prin,
                 accuracy=accuracy, test_data=test_data, lamb = lamb,stop_threshhold=stop_threshhold, stop_criterion=stop_criterion)

        beta_temp = self.beta
        weights_temp, biases_temp = self.beta_to_wb(beta_temp)

        self.weights = weights_temp
        self.biases = biases_temp


    def gradient(self, x, y):
        """ Create the gradient to be used in SGD"""
        mini_batch = zip(x,y)
        grad_beta = list(np.zeros(len(self.beta)))

        for (xi,yi) in mini_batch:
            grad = self.gradient_data_point(xi,yi)

            for i, el in enumerate(grad_beta):
                grad_beta[i] += grad[i]

        return grad_beta


    def gradient_data_point(self, x, y):
        """
        Calculates the gradient of the network's parameter vector (wrt the
        cost function), to be used by the SGD function of the Regression
        class. Uses the backpropagation algorithm. Takes in a set of
        input and output data (in practice a mini-batch).
        """

        beta_temp = self.beta
        weights_temp, biases_temp = self.beta_to_wb(beta=beta_temp)


        nabla_b = [np.zeros(b.shape) for b in biases_temp]
        nabla_w = [np.zeros(w.shape) for w in weights_temp]

        activation = x
        activations = [x]
        zs = []
        k = -1
        for b, w in zip(biases_temp, weights_temp):

            k += 1
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.afs[k].func(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * \
            self.afs[-1].prime(zs[-1])


        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.n_layers):
            z = zs[-l]
            sp = self.afs[-l].prime(z)
            delta = np.dot(weights_temp[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return self.wb_to_beta(weights=nabla_w, biases=nabla_b)


    def wb_to_beta(self, weights=None, biases=None):
        """
        Ravels vectors of weights and biases to one beta vector
        for compatability with Regression's SGD method
        """

        if isinstance(weights, list): all_weights = weights
        else: all_weights = self.weights

        if isinstance(biases, list): all_biases = biases
        else: all_biases = self.biases

        beta_temp = list(np.zeros(self.len_wb))
        k = -1

        for w_matrix in all_weights:
            k += 1
            beta_temp[k] = w_matrix

        for bs in all_biases:
            k += 1
            beta_temp[k] = bs

        return beta_temp

    def beta_to_wb(self, beta=[0]):
        """Unravels beta vector to vectors of weights and biases."""

        all_weights = self.weights
        all_biases = self.biases

        if isinstance(beta,list): beta_temp = list(beta)
        else: beta_temp = list(self.beta)

        for i, w_matrix in enumerate(all_weights):
            all_weights[i] = beta_temp.pop(0)

        for l, bs in enumerate(all_biases):
            all_biases[l] = beta_temp.pop(0)

        return all_weights, all_biases


    def set_afs_to_standard(self):
        """ sets the chosen activation function"""
        afs_temp = []
        afs_prime_temp = []

        for b in self.biases:

            afs_temp.append(self.standard_af)

        self.afs = afs_temp
        self.afs_prime = afs_prime_temp


    def find_len_wb(self):
        len_wb = 0
        for w_mat in self.weights:
            len_wb += 1
        for b_vec in self.biases:
            len_wb += 1

        return len_wb


    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)



    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        self.test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in self.test_results)



class Sigmoid:
    """
    Activation function class.

    func(z): The actual function value
    prime(z): The derivative
    """
    def func(self,z):
        return 1.0/(1.0 + np.exp(-z))

    def prime(self, z):
        return self.func(z)*(1-self.func(z))



class RELU:
    """
    Activation function class.

    func(z): The actual function value
    prime(z): The derivative
    """
    def func(self,z):
        result = np.zeros(shape=z.shape)

        for i,el in enumerate(z):
            result[i] = max(0,el) + min(0,0.1*el)
        return result

    def prime(self,z):
        result = np.zeros(shape=z.shape)
        for i,el in enumerate(z):
            result[i] = 1*(el>=0) + 0.1*(el<0)

        return result



class Leaky_RELU:
    """
    Activation function class.

    func(z): The actual function value
    prime(z): The derivative
    """
    def __init__(self, a):
        self.a = a

    def func(self,z):
        result = np.zeros(shape=z.shape)

        for i,el in enumerate(z):
            result[i] = max(0,el) + min(0,self.a*el)

        return result


    def prime(self,z):
        result = np.zeros(shape=z.shape)
        for i,el in enumerate(z):
            result[i] = 1*(el>=0) + self.a*(el<0)
        return result



class Softmax:
    """
    Activation function class.

    func(z): The actual function value
    prime(z): The derivative
    """
    def func(self, z):
        numerator = np.exp(z)
        denominator = sum(np.exp(z))

        return numerator/denominator

    def prime(self, z):
        return self.func(z)*(1 - self.func(z))


class Linear:
    """
    Activation function class.

    func(z): The actual function value
    prime(z): The derivative
    """
    def func(self, z):
        return z

    def prime(self, z):
        return 1


class Binary:
    """
    Activation function class.

    func(z): The actual function value
    prime(z): The derivative
    """
    def func(self, z):
        return 1*(z>0)

    def prime(self, z):
            return 0


class OLS(Regression):
    """
    Class for OLS regression
    fit(X, y): Find the OLS regression parameters beta and the variance of the
               fitted parameters.

    gradient(X, y): Calculate the gradient of the OLS cost function
    """
    def fit(self, X, y):
        self.beta = np.linalg.pinv(X) @ y

        self.residual(X,y)
        bv = np.sqrt(self.sigma_hat * tools.SVDinv(X.T @ X).diagonal())
        self.beta_var = bv.ravel()

    def gradient(self, X, y):
        n = X.shape[0]
        return (2.0/n)*X.T @ (X @ self.beta -  y)

class Ridge(Regression):
    """ Class for Ridge regression, takes lambda value as input

    fit(X, y): Find the OLS regression parameters beta and the variance of the
               fitted parameters.

    gradient(X, y): Calculate the gradient of the OLS cost function
    """
    def __init__(self, lamb):
        self.lamb = lamb

    def fit(self, X, y):
        lamb=self.lamb
        I = np.eye(X.shape[1])  # Identity matrix - (p,p)
        self.beta = np.linalg.pinv( X.T @ X + lamb*I) @ X.T @ y

        self.residual(X,y)
        a = np.linalg.pinv(X.T @ X + lamb*I)
        bv = np.sqrt(self.sigma_hat * (a @ (X.T @ X) @ a.T).diagonal())
        self.beta_var = bv.ravel()

    def gradient(self,X,y):
        n = X.shape[0]
        return (2.0/n)*X.T @ (X @ (self.beta) - y) + 2*self.lamb*self.beta



#==============================================================================
# Helper functions

def split_data(data, keep=False):
    x = []
    y = []

    for tupl in data:
        x.append(tupl[0])
        y.append(tupl[1])

    data = zip(x,y)

    if keep: return x,y, data
    else: return x,y


def L1(w):
    return np.sign(w)

def L2(w):
    return w


def format_nxp(data_points):
    new_data_points = list(np.zeros(len(data_points)))

    for i, data_point in enumerate(data_points):
        new_data_point = list(np.zeros(len(data_point)))

        for j, parameter_value in enumerate(data_point):
            new_data_point[j] = [parameter_value]

        new_data_points[i] = np.array(new_data_point)

    return new_data_points



def format_nx1(data_points):
    new_data_points = list(np.zeros(len(data_points)))

    for i, data_point in enumerate(data_points):
        new_data_points[i] = np.array([[data_point]])

    return new_data_points


def check_if_stop(ep, max_ep, mse=None, threshhold = 5):
    if ep > max_ep-1:
        return True
    elif mse==None or threshhold == None:
        return False
    elif ep < 2*threshhold:
        return False
    else:
        mean_now = np.mean(mse[ep-threshhold:ep])
        mean_before = np.mean(mse[ep-2*threshhold:ep-threshhold])

        return (mean_before*0.99<mean_now)


#==============================================================================
# Helper functions


def test_OLS():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, r2_score
    np.random.seed(42)

    # setup data
    x1 = np.arange(0, 1, 0.05); x2 = np.arange(0, 1, 0.05)
    x1, x2 = np.meshgrid(x1, x2)
    y = x1.ravel()**2 + x2.ravel()**2
    noise = 0.1*np.random.normal(0, 1, y.shape)
    y = y + noise
    input = np.c_[x1.ravel(), x2.ravel()]

    # Design matrix
    X = PolynomialFeatures(degree=10).fit_transform(input)

    # The OLS class
    model = OLS()
    model.fit(X, y)
    coef = model.beta
    y_pred = model.predict(X)
    r2 = model.r2score(X,y)
    mse = model.mse(X,y)

    # Using Scikitlearn
    model_sklearn = LinearRegression(fit_intercept=False).fit(X, y)
    coef_sklearn = model_sklearn.coef_
    y_pred_sklearn = model_sklearn.predict(X)
    mse_sklearn = mean_squared_error(y, y_pred_sklearn)
    r2_sklearn = r2_score(y, y_pred_sklearn)

    tol = 1e-12
    sucsess_r2 = np.abs(r2_sklearn - r2) < tol
    sucsess_mse = np.abs(mse_sklearn - mse) < tol
    sucsess_coef = np.all(np.abs(coef_sklearn - coef) < tol)
    sucsess_y_pred = np.all(np.abs(y_pred_sklearn - y_pred) < tol)

    assert sucsess_r2, "R2-scores are not the same: {:.8f} vs {:.8f}".format(r2_sklearn, r2)
    assert sucsess_mse, "Mean Squared errors are not the same"
    assert sucsess_coef, "Coefficients are not the same"
    assert sucsess_y_pred, "predicted y-values are not the same"


if __name__=="__main__":
    test_OLS()
    print("Passed sklearn test")
