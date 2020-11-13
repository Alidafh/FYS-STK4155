#!/usr/bin/python
import numpy as np
import tools as tools
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import SGDRegressor, LinearRegression
from regression import OLS, Ridge
import matplotlib.pyplot as plt

#Generate the data for the Franke function
input, y = tools.GenerateDataFranke(ndata=1000, noise_str=0.1)

#-----------------------------------------------------------------------------

d_max = 6
degrees = np.arange(1, d_max+1)

mse_basic = np.zeros((d_max, 2))
r2_basic = np.zeros((d_max, 2))

mse_stochastic = np.zeros((d_max, 2))
r2_stochastic = np.zeros((d_max, 2))

mse_stochastic1 = np.zeros((d_max, 2))
r2_stochastic1 = np.zeros((d_max, 2))

diff_parameter = np.zeros(d_max)

for d in range(d_max):
    print("Degree: ", d)
    X = tools.PolyDesignMatrix(input.T, d=degrees[d])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = tools.scale_X(X_train, X_test, scaler="manual")
    #X_train, X_test = tools.scale_X(X_train, X_test, scaler="sklearn")


    # Basic OLS
    model_basic = OLS()
    model_basic.fit(X_train, y_train)
    y_p_b = model_basic.predict(X_test)
    y_f_b = model_basic.predict(X_train)

    r2_basic[d, 0] =r2_score(y_test, y_p_b)
    r2_basic[d, 1] =r2_score(y_train, y_f_b)

    mse_basic[d, 0] = mean_squared_error(y_test, y_p_b)
    mse_basic[d, 1] = mean_squared_error(y_train, y_f_b)


    # SGD OLS: learn_rate: 0.01
    model_sgd = OLS()
    loss = model_sgd.SGD_v3(X_train, y_train, n_epochs=1000, batch_size=5, learn_rate=0.01, tol=1e-3, shuffle=True)

    y_p = model_sgd.predict(X_test)
    y_f = model_sgd.predict(X_train)

    mse_stochastic[d, 0] = mean_squared_error(y_test, y_p)#model_sgd.mse(X_test, y_test)
    mse_stochastic[d, 1] = mean_squared_error(y_train, y_f)#model_sgd.mse(X_train, y_train)

    r2_stochastic[d, 0] = r2_score(y_test, y_p)#model_sgd.r2score(X_test, y_test)
    r2_stochastic[d, 1] = r2_score(y_train, y_f)#model_sgd.r2score(X_train, y_train)

    # Using SGD from sklearn
    #np.random.seed(42)
    beta_0 = np.random.randn(X_train.shape[1])
    sgdreg1 = SGDRegressor(max_iter = 1000, eta0=0.01, learning_rate="constant", shuffle=True, random_state=42, tol=1e-3)
    sgdreg1.fit(X_train, y_train)#, coef_init=beta_0)
    print("    ", sgdreg1.n_iter_)
    y_pred = sgdreg1.predict(X_test)
    y_fit = sgdreg1.predict(X_train)

    r2_stochastic1[d, 0] = r2_score(y_test, y_pred)
    r2_stochastic1[d, 1] =r2_score(y_train, y_fit)

    mse_stochastic1[d, 0] = mean_squared_error(y_test, y_pred)
    mse_stochastic1[d, 1] = mean_squared_error(y_train, y_fit)


plt.figure()
plt.grid()
plt.plot(degrees, r2_basic[:,0], label="OLS regression")
plt.plot(degrees, r2_stochastic[:,0], label="SGD regression (lr=0.01, M=5)")
plt.plot(degrees, r2_stochastic1[:,0], label="Scikit SGDregresson")
plt.legend()
plt.xlabel("Model complexity")
plt.ylabel("R2-score")
plt.show()
