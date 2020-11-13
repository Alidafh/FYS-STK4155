#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 08:57:07 2020

@author: gert and alida
"""


#!/usr/bin/python
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
        """ Get the residuals, RSS, ndf and sigma_hat"""
        y_predict = self.predict(X)
        self.residuals = y_predict - y
        self.residual_sum_squares = sum(self.residuals**2)
        self.ndf = len(y)  - len(self.beta)
        self.loss = self.residual_sum_squares/len(y)
        self.sigma_hat = self.residual_sum_squares/self.ndf

    def mse(self, X, y):
        y_predict = self.predict(X)
        mse = np.mean(np.mean((y - y_predict)**2))
        return mse

    def r2score(self, X, y):
        y_predict = self.predict(X)
        r2score = 1 - ((np.sum((y - y_predict)**2))/(np.sum((y - np.mean(y))**2)))
        return r2score

    def Bootstrap(self, X, y, nbs, ts=0.2):
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

    def GD(self, X, y, maxiter, learn_rate):
        n = X.shape[0]
        p = X.shape[1]

        self.loss = np.zeros(maxiter)
        self.epochs = np.zeros(maxiter)
        self.beta = np.random.randn(p)
        iter = 0
        while iter < maxiter:
            gradient = self.gradient(X, y)
            step_size = learn_rate*gradient
            self.beta = self.beta - step_size
            self.residual(X, y)
            self.loss[iter] = (1/(n-p))*self.residual_sum_squares
            self.epochs = iter
            iter += 1

    @nb.jit(forceobj=True)
    def SGD(self, X, y, n_epochs=50, batch_size=5, learn_rate=None, gamma=None, prin=False,
            classification=False, test_data=None, lamb = 0, tol=None):
        """
        Fit the linear model by minimization using Stochastic Gradient Descent
        """
        n = len(X)              # number of datapoints
        p = len(X[0])           # number of parameters
        m = int(n/batch_size)   # number of minibatches

        np.random.seed(42)
        if self.beta == None: self.beta = np.random.randn(p)
    
        momentum = list(np.zeros(p))
        step_size = list(np.zeros(p))
        loss_ = list(np.zeros(n_epochs))

        loss_test = list(np.zeros(n_epochs))

        for ep in range(n_epochs):
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

                for k, el in enumerate(beta_temp):
                    momentum[k] = gamma*momentum[k] + (1 - gamma)*gradient[k] if gamma else gradient[k]
#                    lr = learn_rate if learn_rate else tools.learning_schedule(ep*m+i, 5, 50)
                    lr = learn_rate if learn_rate else tools.learning_schedule(ep+i, 5, 50)
                    step_size[k] = momentum[k]*lr
                    beta_temp[k] = el - step_size[k]/len_batch - self.regularize_index(k)*self.regularization_type(el)*lamb*lr/n

                self.beta = beta_temp

            self.residual(X,y)
            loss_[ep] = np.sum(self.loss)

            info = "Epoch: {:}/{:} \t Loss: {:.8f}".format(ep+1, n_epochs, loss_[ep])
            if prin:
                print(info+"  ", lr)

            if test_data and prin:
                # xt, yt = split_data(test_data)
                # yr = [np.zeros(10) for y in yt]
                # for y in yt:
                #     yr[y] = 1

                # self.residual(xt,yr)
                # loss_test[ep] = np.sum(self.loss)
                more_info = " \t Accuracy: {1} / {2}".format(
                ep, self.evaluate(test_data), len(test_data))
                info = info + more_info

            # check progress to see if the new loss is the same as the last loss
            if tol:
                if np.abs(loss_[ep-1] - loss_[ep]) < tol:
                    print("Converged after {:} iterations/epochs".format(ep+1))
                    print()
                    break

        if prin:
            print()
            print("*************************************!")

        if not classification:
            self.loss_SGD = loss_
            return loss_



class Network(Regression):    # "(object)" in book


    def __init__(self, layer_sizes):

        self.n_layers = len(layer_sizes)

        self.layer_sizes = layer_sizes

        self.biases = [np.random.randn(y,1) for y in layer_sizes[1:]]  # First layer is input, thus no biases.

        self.weights = [np.random.randn(y,x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]  # Vector of matrices, where columns are the node within the erlier layer, rows the node in the affected layer, and values are weights.

        self.len_wb = self.find_len_wb()

        self.beta = self.wb_to_beta()

        self.standard_af = Sigmoid()

        self.afs = None

        self.set_afs_to_standard()

        self.regularization_type = L2

        self.regularize_indices = [i<len(self.weights) for i in range(self.len_wb)]



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



    def train(self, x, z, n_epochs=30, batch_size=10, learn_rate=3, gamma=None, prin=None, classification=True, test_data=None,
              lamb = 0):

        """
        The function that trains the neural network. In practice takes the
        parameters that have been optimized by the SGD function and uses them
        to update the network.
        """
        print("NN, train (classification): ", classification)

        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("optimize() is not implemented yet")
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self.SGD(x,z, n_epochs=n_epochs, batch_size=batch_size, learn_rate=learn_rate,
                gamma=gamma, prin=prin, classification=classification, test_data=test_data, lamb = lamb)

        beta_temp = self.beta
        weights_temp, biases_temp = self.beta_to_wb(beta_temp)


        self.weights = weights_temp
        self.biases = biases_temp




    def gradient(self, x, y):
        print("NN, gradient (x)", type(x), np.shape(x))
        print("NN, gradient (y)", type(y), np.shape(y))

        mini_batch = zip(x,y)

        grad_beta = list(np.zeros(len(self.beta)))

        print("NN, gradient (loop)")
        for (xi, yi) in mini_batch:
            grad = self.gradient_data_point(xi,yi)
            for i, el in enumerate(grad_beta):
                print("      ", i, el)
                grad_beta[i] += grad[i]

        return grad_beta




    def gradient_data_point(self, x, y):

        """
        Calculates the gradient of the network's parameter vector (wrt the
        cost function), to be used by the SGD function of the Regression
        class. Uses the backpropagation algorithm. Takes in a set of
        input and output data (in practice a mini-batch).

        EDIT:

        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        ValueError: shapes (1,1) and (28,) not aligned: 1 (dim 1) != 28 (dim 0)

        """
        print("NN, gradient_data_point (x)", type(x), np.shape(x))
        print("NN, gradient_data_point (y)", type(y), np.shape(y))


        beta_temp = self.beta
        weights_temp, biases_temp = self.beta_to_wb(beta=beta_temp)
        print("NN, gradient_data_point (beta_temp)", np.shape(beta_temp))
        print("NN, gradient_data_point (weights_temp)", np.shape(weights_temp))
        print("NN, gradient_data_point (biases_temp)", np.shape(biases_temp))

        nabla_b = [np.zeros(b.shape) for b in biases_temp]
        nabla_w = [np.zeros(w.shape) for w in weights_temp]
        print("NN, gradient_data_point (nabla_b)", np.shape(nabla_b))
        print("NN, gradient_data_point (nabla_w)", np.shape(nabla_w))
        print()
        activation = x
        activations = [x]
        zs = []
        k = -1
        for b, w in zip(biases_temp, weights_temp):
            print("NN, gradient_data_point (b in loop)", type(b), np.shape(b))
            print("NN, gradient_data_point (w in loop)", type(w), np.shape(w))

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
        Ravels vectors  of weights and biases to one beta vector
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


        """
        Unravels beta vector to vectors of weights and biases.
        """

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
        # for (x,y) in test_data: print(y)
        return sum(int(x == y) for (x, y) in self.test_results)



class Sigmoid:

    def func(self,z):

        return 1.0/(1.0 + np.exp(-z))

    def prime(self, z):
        return self.func(z)*(1-self.func(z))



class RELU:

    def func(self,z):

        result = np.zeros(shape=z.shape)

        for i,el in enumerate(z):
            result[i] = max(0,el)
        return result

    def prime(self,z):
        result = np.zeros(shape=z.shape)
        for i,el in enumerate(z):
            result[i] = 1*(el>=0)

        return result



class Leaky_RELU:

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
            result[i] = 1*(el>=0) -self.a*(el<0)
        return result



class Softmax:

    def func(self, z):
        numerator = np.exp(z)
        denominator = sum(np.exp(z))

        return numerator/denominator

    def prime(self, z):
        return self.func(z)*(1 - self.func(z))


class No:

    def func(self, z):
        return z

    def prime(self, z):
        return 1



class Binary:

    def func(self, z):
        return 1*(z>0)

    def prime(self, z):
            return 0


class OLS(Regression):
    """ Class for OLS regression"""
    def fit(self, X, y):
        self.beta = np.linalg.pinv(X) @ y

        self.residual(X,y)
        bv = np.sqrt(self.sigma_hat * tools.SVDinv(X.T @ X).diagonal())
        self.beta_var = bv.ravel()

    def gradient(self, X, y):
        if isinstance(X, list):
            X = np.array(X)

        n = X.shape[0]
        #print("OLS:  ", n)
        #print("OLS:  ", (2.0/n)*X.T @ (X @ self.beta -  y))
        #print("OLS:  ", self.beta)
        return (2.0/n)*X.T @ (X @ self.beta -  y)

class Ridge(Regression):
    """ Class for Ridge regression, takes lambda value as input"""
    def __init__(self, lamb):
        self.lamb = lamb
        super().__init__()

    def fit(self, X, y):
        lamb=self.lamb
        I = np.eye(X.shape[1])  # Identity matrix - (p,p)
        self.beta = np.linalg.pinv( X.T @ X + lamb*I) @ X.T @ y

        self.residual(X,y)
        a = np.linalg.pinv(X.T @ X + lamb*I)
        bv = np.sqrt(self.sigma_hat * (a @ (X.T @ X) @ a.T).diagonal())
        self.beta_var = bv.ravel()

    def gradient(self,X,y):
        if isinstance(X, list):
            X = np.array(X)

        n = X.shape[0]
        return (2.0/n)*X.T @ (X @ (self.beta) - y) + 2*self.lamb*self.beta




#==============================================================================

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


def test_Regression():
    X, y = tools.GenerateDataLine(100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test = tools.scale_X(X_train, X_test)

    model = OLS()
    model.fit(X_train, y_train)
    print("Beta : ", model.beta, sep='\n')
    print("r2 : ", model.r2score(X_test, y_test), '\n')

    model2 = OLS()
    model2.GD1(X_train, y_train, maxiter=1000, learn_rate=0.1)
    print("Beta : ", model2.beta, sep='\n')
    print("r2 : ", model2.r2score(X_test, y_test), '\n')
if __name__=="__main__":
    #test_OLS()
    #print("Passed sklearn test")
    test_Regression()







def split_data(data):

    x = []
    y = []

    for tupl in data:
        x.append(tupl[0])
        y.append(tupl[1])

    return x, y


def L1(w):
    return np.sign(w)

def L2(w):
    return w
