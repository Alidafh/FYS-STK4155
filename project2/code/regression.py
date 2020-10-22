import numpy as np
import tools as tools
from sklearn.model_selection import train_test_split

class Regression:
    def __init__(self):
        self.X = None
        self.y = None
        self.beta = None
        self.beta_var=None
        self.lamb = None

    def predict(self, X):
        print("PREDICT   ")
        print("X:  ", X.shape)
        print()
        y_predict = X @ self.beta
        return y_predict

    def mse(self, X, y):
        print("MSE:  ")
        y_predict = self.predict(X)
        mse = np.mean(np.mean((y - y_predict)**2))
        return mse

    def r2score(self, X, y):
        print("R2     ")
        y_predict = self.predict(X)
        r2score = 1 - ((np.sum((y-y_predict)**2))/(np.sum((y-np.mean(y))**2)))
        return r2score

class OLS(Regression):
    def fit(self, X, y):
        print("FIT   ")
        self.beta = np.linalg.pinv(X.T @ X) @ X.T @ y

        y_predict = self.predict(X)
        residuals = y - y_predict
        residual_sum_squares = residuals.T @ residuals
        lower = len(y) - len(self.beta)
        sigma_hat = residual_sum_squares/lower
        self.beta_var = np.sqrt(sigma_hat * tools.SVDinv(X.T @ X).diagonal())

class Ridge(Regression):
    def fit(self, X, y, lamb):
        I = np.eye(X.shape[1])  # Identity matrix - (p,p)
        self.beta = np.linalg.pinv( X.T @ X + lamb*I) @ X.T @ y

        y_predict = self.predict(X)
        residuals = y - y_predict
        residual_sum_squares = residuals.T @ residuals
        lower = len(y) - len(self.beta)
        sigma_hat = residual_sum_squares/lower
        a = np.linalg.pinv(X.T @ X + lamb*I)
        self.beta_var = np.sqrt(sigma_hat * (a @ (X.T @ X) @ a.T).diagonal())

#==============================================================================
def test_OLS():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, r2_score
    np.random.seed(42)

    # setup data
    x1 = np.arange(0, 1, 0.1); x2 = np.arange(0, 1, 0.1)
    x1, x2 = np.meshgrid(x1, x2)
    y = x1.ravel()**2 + x2.ravel()**2
    noise = 0.1*np.random.normal(0, 1, y.shape)
    y = y + noise
    input = np.c_[x1.ravel(), x2.ravel()]

    # Design matrix
    X = PolynomialFeatures(degree=3).fit_transform(input)

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

    assert sucsess_r2, "R2-scores are not the same"
    assert sucsess_mse, "Mean Squared errors are not the same"
    assert sucsess_coef, "Coefficients are not the same"
    assert sucsess_y_pred, "predicted y-values are not the same"

if __name__=="__main__":
    test_OLS()
    print("Passed sklearn test")
