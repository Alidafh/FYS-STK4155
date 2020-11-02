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

    def predict(self, X):
        """Get the predicted value"""
        y_predict = X @ self.beta
        return y_predict

    def residual(self, X, y):
        """ Get the residuals, RSS, ndf and sigma_hat"""
        y_predict = self.predict(X)
        self.residuals = y_predict - y
        self.residual_sum_squares = self.residuals.T @ self.residuals
        self.ndf = len(y) - len(self.beta)
        self.loss = self.residual_sum_squares/self.ndf
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
    def SGD(self, X, y, n_epochs=50, batch_size=5, learn_rate=None, gamma=None, prin=False):
        """Fit the linear model by minimization using Stochastic Gradient Descent

        Find beta values for OLS or Ridge using the SGD method. The method also
        creates a logfile of the loss for each epoch number and batch number,
        filename starts with SGDLOG_.

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

        (prin : bool, default=False
            If set to True a progress report is printed of the loss for each
            epoch and batch number.)

        Attributes:
        ----------
        beta : ndarray, shape (p,)
            Array of regression parameters

        loss_ep : ndarray, shape(n_epochs,)
            The loss as a function of epoch

        """
        n = X.shape[0]          # number of datapoints
        p = X.shape[1]          # number of parameters
        m = int(n/batch_size)   # number of minibatches

        np.random.seed(42)
        self.beta = np.random.randn(p)
        momentum = np.zeros(p)
        loss_ = np.zeros((m, n_epochs))

        for ep in range(n_epochs):
            #np.random.shuffle(X)    # added
            for i in range(m):
                random_index = np.random.randint(m)
                xi = X[random_index:random_index+batch_size]
                yi = y[random_index:random_index+batch_size]
                gradient = self.gradient(xi, yi)
                momentum = gamma*momentum + (1 - gamma)*gradient if gamma else gradient
                #step_size = momentum*learn_rate if learn_rate else momentum*tools.learning_schedule(ep*m+i, batch_size, n_epochs)
                step_size = momentum*learn_rate if learn_rate else momentum*tools.learning_schedule(ep*m+i, 5, 50)
                self.beta -= step_size

                self.residual(X,y)
                loss_[i, ep] = self.loss

                info = "Epoch: {:}/{:} Batch: {:}/{:} Loss: {:.3f}".format(ep+1, n_epochs, i+1, m, loss_[i, ep])
                print(info)
        print()

        self.loss_SGD = loss_
        return loss_


class OLS(Regression):
    """ Class for OLS regression"""
    def fit(self, X, y):
        self.beta = np.linalg.pinv(X) @ y

        self.residual(X,y)
        bv = np.sqrt(self.sigma_hat * tools.SVDinv(X.T @ X).diagonal())
        self.beta_var = bv.ravel()

    def gradient(self, X, y):
        n = X.shape[0]
        return (2.0/n)*X.T @ (X @ self.beta -  y)

class Ridge(Regression):
    """ Class for Ridge regression, takes lambda value as input"""
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


class SGD(Regression):
    def __init__(self, model):
        super().__init__(self)
        self.model = model

    def GD1(self, X, y, maxiter, learn_rate):
        n = X.shape[0]
        p = X.shape[1]

        self.loss = np.zeros(maxiter)
        self.epochs = np.zeros(maxiter)
        self.beta = np.random.randn(p)
        iter = 0
        while iter < maxiter:
            gradient = self.gradient(X, y)
            #gradient = model.gradient(X, y)
            step_size = learn_rate*gradient
            self.beta = self.beta - step_size
            self.residual(X, y)
            self.loss[iter] = (1/(n-p))*self.residual_sum_squares
            self.epochs = iter
            iter += 1

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
