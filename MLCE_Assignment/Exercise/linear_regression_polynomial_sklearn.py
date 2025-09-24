import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# define function we want to approximate
def my_fun(x):
#    y = x**2*np.sin(x)
    y = x**3
    return y

## sample from function we want to approximate
n_train = 10
x_train = ((np.random.rand((n_train))-0.5)*10).reshape(-1, 1)
y_train = my_fun(x_train) + 1*np.random.randn(n_train).reshape(-1, 1)

# evaluate function to be approximated for plotting
n_pred = 100
x_pred = np.linspace(-5, 5, n_pred).reshape(-1, 1)
y_plot = my_fun(x_pred)

# transform the input features to polynomial features
poly_features = PolynomialFeatures(degree=5)
x_train_poly = poly_features.fit_transform(x_train)

# create and fit the polynomial regression model 
model = LinearRegression()
model.fit(x_train_poly, y_train)

# predict the test input features 
x_pred_poly = poly_features.fit_transform(x_pred)
y_pred = model.predict(x_pred_poly)

# plot function to be approximated, input and test data, 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_pred,y_plot,color='lightblue')
ax.scatter(x_train,y_train,color='red',marker='+')
ax.plot(x_pred,y_pred,color='lightgreen')
ax.set(title='linear regression',ylabel='y',xlabel='x')
ax.legend(["org fun", "samples", "LR pred"], loc ="best")

plt.show()