import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parameters
gamma = 1.0  # gyromagnetic ratio
alpha = 0.1  # damping factor
H = np.array([0.0, 0.0, 1.0])  # external magnetic field

def llg(y, t):
    m = y.reshape((5, 3))  # reshape flat array back to 5 vectors of 3 components
    dm_dt = np.empty_like(m)
    for i in range(5):
        dm_dt[i] = -gamma * np.cross(m[i], H) - alpha * np.cross(m[i], np.cross(m[i], H))
    return dm_dt.flatten()  # return as flat array

# Initial conditions: 5 spins aligned with the z-axis
y0 = np.tile([0.0, 0.0, 1.0], 5)

# Time grid
t = np.linspace(0, 10, 1000)

# Solve ODE
y = odeint(llg, y0, t)

# Plot results
for i in range(5):
    plt.figure(figsize=(10, 6))
    plt.plot(t, y[:, 3*i], label='mx')
    plt.plot(t, y[:, 3*i+1], label='my')
    plt.plot(t, y[:, 3*i+2], label='mz')
    plt.legend()
    plt.title(f'Spin {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Magnetization')
    plt.grid(True)
    plt.show()