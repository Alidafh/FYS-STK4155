import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from regression import OLS, Ridge

np.random.seed(42)
# ------------------------------------------------------------------------------
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    z = term1 + term2 + term3 + term4
    return z

##############################################################################

# Set up data
x1 = np.arange(0, 1, 0.1)
x2 = np.arange(0, 1, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = FrankeFunction(x1, x2).ravel()
noise = 0.1*np.random.normal(0, 1, y.shape)
y = y + noise

# set up design matrix
input = np.c_[x1.ravel(), x2.ravel()]
X = PolynomialFeatures(degree=4).fit_transform(input)

##############################################################################

# SKLEARN
model_sklearn = LinearRegression(fit_intercept=False).fit(X, y)
coef_sklearn = model_sklearn.coef_
y_pred_sklearn = model_sklearn.predict(X)
mse_sklearn = mean_squared_error(y, y_pred_sklearn)
r2_sklearn = r2_score(y, y_pred_sklearn)

print("--------------------------------")
print("SKLEARN")
print("--------------------------------")
print('coefficients:', coef_sklearn, sep='\n')
print('Mean squared error:', mse_sklearn, sep='\n')
print('Coefficient of determination:', r2_sklearn, sep='\n')
print('predicted response:', y_pred_sklearn, sep='\n')
print()

##############################################################################

# Egen klasse
model = OLS()
model.fit(X, y)
coef = model.beta
y_pred = model.predict(X)
r2 = model.r2_score(X,y)
mse = model.mse(X,y)

print("--------------------------------")
print("MANUELL")
print("--------------------------------")
print('coefficients:', coef, sep='\n')
print('Mean squared error:', mse, sep='\n')
print('Coefficient of determination:', r2, sep='\n')
print('predicted response:', y_pred, sep='\n')
print()
##############################################################################
# Comparison

diff_coef = np.abs(coef_sklearn - coef)
diff_r2 = np.abs(r2_sklearn - r2)
diff_mse = np.abs(mse_sklearn - mse)
diff_y_pred = np.abs(y_pred_sklearn - y_pred)

print("--------------------------------")
print("comparison sklearn - manuell")
print("--------------------------------")
print('coefficients:', np.array2string(diff_coef, formatter={'float_kind':lambda x: "%.8f" % x}), sep='\n')
print('Mean squared error: %.8f' %diff_mse, sep='\n')
print('Coefficient of determination: %.8f'% diff_r2, sep='\n')
print('predicted response:', np.array2string(diff_y_pred, formatter={'float_kind':lambda x: "%.8f" % x}), sep='\n')
print()
