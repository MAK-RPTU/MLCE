import numpy as np
import matplotlib.pyplot as plt

# implement LTIclass
class LTIsystem():
    def __init__(self, A, B) -> None:
        self.A = A
        self.B = B

    def _step(self, x, u) -> np.ndarray:
        x_next = self.A @ x + self.B @ u
        return x_next

# define the LTI system we want to simulate
Ts = 0.1
A = np.array([[1, Ts], [0, 1]])
B = np.array([[0],[Ts]])

# create LTI system
lin_sys = LTIsystem(A,B)

# select a controller gain
K = np.array([[-0.2, -0.2]])

# simulation parameter
x_0 = np.array([[2], [1]])
T_sim = 500
u = np.array([[1]])
x = x_0
x_trajectory = np.zeros((2,T_sim))
u_trajectory = np.zeros((1,T_sim))

# simulate the system
for t in range(0, T_sim):
    u = K @ x
    x_next = lin_sys._step(x,u)
    x = x_next
#    x_trajectory = np.append(x_trajectory,x_next,axis=1)
#    u_trajectory = np.append(u_trajectory,u,axis=1)
    x_trajectory[:,t] = x_next.reshape(2,)
    u_trajectory[:,t] = u.reshape(1,)

# plot results
t_plot = np.arange(0,T_sim)

fig, ax = plt.subplots(1, 2, layout="constrained")
ax[0].plot(t_plot, x_trajectory[0,0:T_sim])
ax[0].plot(t_plot, x_trajectory[1,0:T_sim])
ax[0].set(ylabel='x', xlabel='t')
ax[0].legend(['x_1','x_2'])

ax[1].plot(t_plot, u_trajectory[0,:])
ax[1].set(ylabel='u', xlabel='t')
ax[1].legend(['u'])

plt.show()