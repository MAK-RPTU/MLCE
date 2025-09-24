import numpy as np
import matplotlib.pyplot as plt

# define function we want to approximate
def my_fun(x):
    y = x**2*np.sin(x)
    return y

## sample from function we want to approximate
n_train = 10
x_train = (np.random.rand((n_train))-0.5)*10
print(x_train)
y_train = my_fun(x_train) + 0.1*np.random.randn(n_train)

# evaluate function to be approximated for plotting
n_pred = 100
x_pred = np.linspace(-5,5,n_pred)
y_plot = my_fun(x_pred)

## feature matrix
def feature_matrix_poly(x,M):
    n = np.size(x)
    Phi = np.zeros((n,M))
    for i in range(n):
        for j in range(M):
            Phi[i,j] = x[i]**j
    return Phi

# select order of polynomial
M = 5

# build feature matrix
Phi_train = feature_matrix_poly(x_train,M)

# compute weights
theta_ml = np.linalg.inv(Phi_train.T@Phi_train)@Phi_train.T@y_train.T

# prediction
Phi_pred = feature_matrix_poly(x_pred,M) # build feature matrix
y_pred_LR = Phi_pred@theta_ml            # predict values

# plot figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_pred,y_plot,color='lightblue')
ax.scatter(x_train,y_train,color='red',marker='+')
ax.plot(x_pred,y_pred_LR,color='lightgreen')
ax.set(title='linear regression',ylabel='y',xlabel='x')
ax.legend(["org fun", "samples", "LR pred"], loc ="best")

plt.show()