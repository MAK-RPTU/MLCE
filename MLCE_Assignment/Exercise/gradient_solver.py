import numpy as np
import matplotlib.pyplot as plt


# define QP problem
Q = np.array([[4, 0], [0, 2]])
f = np.array([[1], [2]])

# cost QP problem
def cost(x, Q, f) -> np.ndarray:
    J = 0.5*x.T@Q@x + f.T@x
    return J

# gradient QP problem
def gradient(x, Q, f) -> np.ndarray:
    grad = Q@x + f
    return grad

# compute analytical solution
x_analytical = -np.linalg.inv(Q)@f 
print('analytical sol: ', x_analytical)

# gradient algorithm 
k_max = 100         # max iterations
step_size = 0.02   # step size
x = np.array([[5], [5]]) # initial condition
x_trajectory = np.zeros((2,k_max))  # initialize trajectory optimizer
J_trajectory = np.zeros((1,k_max)) # initialize trajectory cost

# run gradient algorithm
for k in range(0,k_max):
    x_next = x - step_size*gradient(x, Q, f) #  gradient step
    x = x_next                               #  update iterate x
    x_trajectory[:,k] = x.reshape(2,)        # store iterate in trajectory
    J_trajectory[:,k] = cost(x, Q, f)        # store cost interate in trajectory

# plot results
k_plot = np.arange(0,k_max) # define vector for plotting

fig, ax = plt.subplots(1, 2, layout="constrained")
ax[0].plot(k_plot, x_trajectory[0,0:k_max])
ax[0].plot(k_plot, x_trajectory[1,0:k_max])
ax[0].set(ylabel='x', xlabel='k')
ax[0].legend(['x_1','x_2'])

ax[1].plot(k_plot, J_trajectory[0,:])
ax[1].set(ylabel='J', xlabel='k')
ax[1].legend(['J'])

plt.show()

### plot contour plot
x_plot = np.linspace(-5,5,100)      # define x axis
y_plot = np.linspace(-5,5,100).T    # define y axis

J_plot = np.zeros((100,100))        # init cost plot

# loop through space and evaluate cost
for i in range(100):
    for j in range(100):
        J_plot[i,j] = cost(np.append(x_plot[j],y_plot[i]), Q, f)

# plotting
plt.figure()
cs = plt.contour(x_plot, y_plot, J_plot, levels = [0, 0.5, 1, 5, 10, 25])
plt.plot(x_trajectory[0,:], x_trajectory[1,:],'.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('gradient algorithm')
plt.show()
