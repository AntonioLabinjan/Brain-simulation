import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- PARAMETERS ---
T = 200           # total time (ms)
dt = 1            # time step (ms)
time = np.arange(0, T+dt, dt)

N = 100
V_rest = -65
V_reset = -70
V_th = -50
R = 10
tau = 10

# Connectivity (random excitatory/inhibitory)
np.random.seed(42)
W = np.random.randn(N, N) * 0.5
np.fill_diagonal(W, 0)

# Input currents
I_mean = 1.5
I = I_mean + 0.5*np.random.randn(N, len(time))

# Membrane potentials
V = np.ones(N) * V_rest
spikes = np.zeros((N, len(time)))

# Precompute spikes
for t_idx, t in enumerate(time):
    for n in range(N):
        I_syn = np.sum(W[:, n] * spikes[:, t_idx-1] if t_idx>0 else 0)
        dV = (-(V[n]-V_rest) + R*(I[n, t_idx] + I_syn)) / tau
        V[n] += dV*dt

        if V[n] >= V_th:
            V[n] = 30
            spikes[n, t_idx] = 1
            V[n] = V_reset

# --- 2D NEURON POSITIONS ---
grid_size = int(np.sqrt(N))
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
x = x.flatten()
y = y.flatten()

# --- ANIMATION SETUP ---
fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(-1, grid_size)
ax.set_ylim(-1, grid_size)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('black')
ax.set_title("2D Neuron Network Simulation", color='white')

# Neuron scatter
scat = ax.scatter(x, y, c='blue', s=50)

# Draw lines for connectivity
lines = []
for i in range(N):
    for j in range(N):
        if W[i,j] != 0:
            line, = ax.plot([], [], color='yellow', alpha=0.1)
            lines.append((i, j, line))

# Update function for animation
def update(frame):
    # Update neuron colors and sizes
    sizes = 50 + 200*spikes[:, frame]  # spike pulses
    colors = np.where(spikes[:, frame]==1, 'red', 'blue')
    scat.set_sizes(sizes)
    scat.set_color(colors)
    
    # Update connection lines (flash if presynaptic neuron spikes)
    for i,j,line in lines:
        if spikes[i, frame]==1:
            line.set_data([x[i], x[j]], [y[i], y[j]])
            line.set_alpha(0.7)
        else:
            line.set_alpha(0.05)
    
    # Return all artists as a single flat list
    all_artists = [scat] + [line for _, _, line in lines]
    return all_artists

# Create animation
ani = FuncAnimation(fig, update, frames=len(time), interval=50, blit=True)
plt.show()
