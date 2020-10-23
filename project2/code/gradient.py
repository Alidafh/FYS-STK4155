
import numpy as np
import tools as tools
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.model_selection import train_test_split
from regression import OLS, Ridge, GradientDesent
import matplotlib.pyplot as plt
np.random.seed(42)

##############################################################################
input, y = tools.GenerateDataFranke(1000, noise_str=0.1)
X = PolynomialFeatures(degree=3).fit_transform(input)

#X, y = tools.GenerateDataLine(ndata=1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test = tools.scale_X(X_train, X_test)
##############################################################################
print("-------", "OLS","-------", sep='\n' )
linreg = OLS()
linreg.fit(X_train, y_train)
print('Own linreg', linreg.beta, sep='\n')

gdreg = OLS()
gdreg.GD(X_train, y_train, maxiter=1000, learn_rate=0.1)
print("Own GD", gdreg.beta, sep='\n')

sgdreg = OLS()
sgdreg.SGD(X_train, y_train, learn_rate = 0.1, n_epochs=100, batch_size=1)
print("Own SGD", sgdreg.beta, sep='\n')

##############################################################################
print("-------", "ridge","-------", sep='\n' )
rrlinreg = Ridge(lamb=0.1)
rrlinreg.fit(X_train, y_train)
print('Own linreg', rrlinreg.beta, sep='\n')

rrgdreg = Ridge(lamb=0.1)
rrgdreg.GD(X_train, y_train, maxiter=1000, learn_rate=0.1)
print("Own GD", rrgdreg.beta, sep='\n')

rrsgdreg = Ridge(lamb=0.1)
rrsgdreg.SGD(X_train, y_train, learn_rate = 0.1, n_epochs=100, batch_size=1)
print("Own SGD", rrsgdreg.beta, sep='\n')

##############################################################################
ols = OLS()
ols.fit(X, y)
y_hat = ols.predict(X)

sd=int(round(np.sqrt(len(y))))
x1 = input[:,0].reshape(sd,sd)
x2 = input[:,1].reshape(sd,sd)
y1 = y.reshape(sd,sd)
y2 = y_hat.reshape(sd,sd)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x1,x2,y1, s=0.5, c="k")
ax.plot_surface(x1,x2,y2)
plt.show()
"""
ypredict = rrlinreg.predict(X_test)
ypredict2 = rrgdreg.predict(X_test)
ypredict3 = rrsgdreg.predict(X_test)

plt.plot()

plt.plot(X_test[:,1], y_test ,'ro')
plt.plot(X_test[:,1], ypredict, "k-", label="linreg")
plt.plot(X_test[:,1], ypredict2, "b-", label="gdreg")
plt.plot(X_test[:,1], ypredict3, "g-", label="sgdreg")
plt.legend()

#plt.axis([0, 2.0, 0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Random numbers ')
plt.show()

"""
"""
# Importing various packages
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

n = 100
x = 2*np.random.rand(n,1)
y = 4+3*x+np.random.randn(n,1)

X = np.c_[np.ones((n,1)), x]
beta_linreg = np.linalg.inv(X.T @ X) @ (X.T @ y)
print(beta_linreg)
sgdreg = SGDRegressor(max_iter = 50, penalty=None, eta0=0.1)
sgdreg.fit(x,y.ravel())
print(sgdreg.intercept_, sgdreg.coef_)



# Importing various packages
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys

# the number of datapoints
n = 100
x = 2*np.random.rand(n,1)
y = 4+3*x+np.random.randn(n,1)

X = np.c_[np.ones((n,1)), x]
# Hessian matrix
H = (2.0/n)* X.T @ X
# Get the eigenvalues
EigValues, EigVectors = np.linalg.eig(H)
print(EigValues)

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
print(beta_linreg)
beta = np.random.randn(2,1)

eta = 1.0/np.max(EigValues)
Niterations = 1000

for iter in range(Niterations):
    gradient = (2.0/n)*X.T @ (X @ beta-y)
    beta -= eta*gradient

print(beta)
xnew = np.array([[0],[2]])
xbnew = np.c_[np.ones((2,1)), xnew]
ypredict = xbnew.dot(beta)
ypredict2 = xbnew.dot(beta_linreg)
plt.plot(xnew, ypredict, "r-")
plt.plot(xnew, ypredict2, "b-")
plt.plot(x, y ,'ro')
plt.axis([0,2.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Gradient descent example')
plt.show()

# Importing various packages
from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor


#input, y = tools.GenerateDataFranke(1000, noise_str=0.1)  # Set up the data
#X = PolynomialFeatures(degree=2).fit_transform(input)

m = 100
x = 2*np.random.rand(m,1)
y = 4+3*x+np.random.randn(m,1)

X = np.c_[np.ones((m,1)), x]
theta_linreg = np.linalg.inv(X.T @ X) @ (X.T @ y)
print("Own inversion")
print(theta_linreg)
sgdreg = SGDRegressor(max_iter = 50, penalty=None, eta0=0.1)
sgdreg.fit(x,y.ravel())
print("sgdreg from scikit")
print(sgdreg.intercept_, sgdreg.coef_)


theta = np.random.randn(2,1)
eta = 0.1
Niterations = 1000


for iter in range(Niterations):
    gradients = 2.0/m*X.T @ ((X @ theta)-y)
    theta -= eta*gradients
print("theta from own gd")
print(theta)

xnew = np.array([[0],[2]])
Xnew = np.c_[np.ones((2,1)), xnew]
ypredict = Xnew.dot(theta)
ypredict2 = Xnew.dot(theta_linreg)


n_epochs = 50
t0, t1 = 5, 50
def learning_schedule(t):
    return t0/(t+t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T @ ((xi @ theta)-yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta - eta*gradients
print("theta from own sdg")
print(theta)

plt.plot(xnew, ypredict, "r-")
plt.plot(xnew, ypredict2, "b-")
plt.plot(x, y ,'ro')
plt.axis([0,2.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Random numbers ')
plt.show()
"""
