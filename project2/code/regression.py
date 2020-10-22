import numpy as np

class OLS:
    def __init__(self):
        self.X = None
        self.y = None
        self.beta = None

    def fit(self, X, y):
        self.beta = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, X):
        y_predict = X @ self.beta
        return y_predict

    def mse(self, X, y):
        y_predict = self.predict(X)
        #mse = np.mean(np.square(y - y_predict))
        mse = np.mean(np.mean((y - y_predict)**2))
        return mse

    def r2_score(self, X, y):
        y_predict = self.predict(X)
        y_mean = np.mean(y)
        upper_sum = np.sum(np.square(y - y_predict))
        lower_sum = np.sum(np.square(y - y_mean))
        #r2score = 1 - upper_sum / lower_sum
        r2score = 1 - ((np.sum((y-y_predict)**2))/(np.sum((y-np.mean(y))**2)))
        return r2score

class Ridge:
    def __init__(self):
        self.X = None
        self.y = None
        self.beta = None
        self.lamb = None

    def fit(self, X, y, lamb):
        I = np.eye(X.shape[1])  # Identity matrix - (p,p)
        self.beta = np.linalg.pinv( X.T @ X + lamb*I) @ X.T @ y

    def predict(self, X):
        y_predict = X @ self.beta
        return y_predict

    def mse(self, x, y):
        y_predict = self.predict(x)
        mse = np.mean(np.square(y - y_predict))
        return mse

    def r2_score(self, X, y):
        y_predict = self.predict(X)
        y_mean = np.mean(y)
        upper_sum = np.sum(np.square(y - y_predict))
        lower_sum = np.sum(np.square(y - y_mean))
        r2score = 1 - upper_sum / lower_sum
        return r2score
