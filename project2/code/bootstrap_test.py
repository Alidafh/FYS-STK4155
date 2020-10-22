import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tools as tools
from regression import OLS, Ridge
from resample import Resample
np.random.seed(42)
# ------------------------------------------------------------------------------
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    z = term1 + term2 + term3 + term4
    return z

# Set up data
x1 = np.arange(0, 1, 0.4)
x2 = np.arange(0, 1, 0.4)
x1, x2 = np.meshgrid(x1, x2)
y = FrankeFunction(x1, x2).ravel()
noise = 0.1*np.random.normal(0, 1, y.shape)
y = y + noise

# set up design matrix
input = np.c_[x1.ravel(), x2.ravel()]
X = PolynomialFeatures(degree=1).fit_transform(input)

##############################################################################
# Egen klasse
print("TOTAL:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
X_train, X_test = tools.scale_X(X_train, X_test)
print("TRAIN:", X_train.shape)
print("TEST:", X_test.shape)
print()

model = OLS()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = model.r2score(X_test, y_test)
mse = model.mse(X_test, y_test)

coef = model.beta
coef_var = model.beta_var

print('Mean squared error: %.6f' %mse, sep='\n')
print('Coefficient of determination:%.6f'%r2, sep='\n')
print()

nbs = 10
test = Resample()
test.Bootstrap(X_train, y_train, nbs, method = OLS())


"""
print("--------------------------------")
print("MANUELL")
print("--------------------------------")
print('coefficients:', np.array2string(coef, formatter={'float_kind':lambda x: "%.2f" % x}), sep='\n')
print('coefficients variance:', np.array2string(coef_var, formatter={'float_kind':lambda x: "%.2f" % x}), sep='\n')
print('Mean squared error: %.6f' %mse, sep='\n')
print('Coefficient of determination:%.6f'%r2, sep='\n')
print('predicted response:', np.array2string(y_pred, formatter={'float_kind':lambda x: "%.2f" % x}), sep='\n')
print()
"""
