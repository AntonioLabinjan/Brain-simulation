import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
T = 200           # total time (ms)
dt = 1            # time step (ms)
time = np.arange(0, T+dt, dt)

# Network parameters
N = 100             # number of neurons
V_rest = -65      # resting potential (mV)
V_reset = -70     # reset potential (mV)
V_th = -50        # spike threshold (mV)
R = 10            # resistance (MΩ)
tau = 10          # membrane time constant (ms)

# Connectivity (random excitatory/inhibitory)
np.random.seed(42)
W = np.random.randn(N, N) * 0.5  # weights matrix
for i in range(N):
    W[i,i] = 0  # no self-connections

# Input currents (random noise)
I_mean = 1.5
I = I_mean + 0.5*np.random.randn(N, len(time))

# Initialize membrane potentials
V = np.ones((N, len(time))) * V_rest
spikes = np.zeros((N, len(time)))

# Simulation loop
for t in range(1, len(time)):
    for n in range(N):
        # Sum of weighted spikes from other neurons at previous step
        I_syn = np.sum(W[:,n] * spikes[:,t-1])
        dV = (-(V[n,t-1]-V_rest) + R*(I[n,t] + I_syn)) / tau
        V[n,t] = V[n,t-1] + dV*dt

        # Spike check
        if V[n,t] >= V_th:
            V[n,t] = 30      # spike peak for visualization
            spikes[n,t] = 1  # record spike
            if t+1 < len(time):
                V[n,t+1] = V_reset

# Plot spikes as raster
plt.figure(figsize=(10,6))
for n in range(N):
    spike_times = time[spikes[n,:]==1]
    plt.vlines(spike_times, n+0.5, n+1.5)
plt.xlabel("Time (ms)")
plt.ylabel("Neuron")
plt.title("Spike Raster Plot of Neuron Network")
plt.ylim(0.5, N+0.5)
plt.show()

# Optional: plot membrane potentials
plt.figure(figsize=(10,6))
for n in range(N):
    plt.plot(time, V[n,:], label=f'Neuron {n+1}')
plt.axhline(V_th, color='red', linestyle='--', label='Threshold')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Membrane Potentials of Neuron Network")
plt.legend()
plt.show()
