import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 200            # total simulation time (ms)
dt = 1             # time step (ms)
time = np.arange(0, T+dt, dt)

# Neuron parameters (LIF model)
V_rest = -65       # resting potential (mV)
V_reset = -70      # reset potential (mV)
V_th = -50         # spike threshold (mV)
R = 10             # resistance (MΩ)
tau = 10           # membrane time constant (ms)

# Input current (constant + noise)
I_mean = 1.5       # nA
I = I_mean + 0.5*np.random.randn(len(time))

# Initialize membrane potential
V = np.zeros(len(time))
V[0] = V_rest

spikes = []

# Simulation loop
for t in range(1, len(time)):
    dV = (-(V[t-1]-V_rest) + R*I[t]) / tau
    V[t] = V[t-1] + dV*dt

    # Check for spike
    if V[t] >= V_th:
        V[t] = 30  # spike peak (for visualization)
        spikes.append(time[t])
        V[t+1 if t+1 < len(time) else t] = V_reset

# Plot results
plt.figure(figsize=(10,5))

plt.subplot(2,1,1)
plt.plot(time, I, color="purple")
plt.ylabel("Input current (nA)")
plt.title("Neuron Input and Spiking Activity")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(time, V, color="black")
plt.axhline(V_th, color="red", linestyle="--", label="Threshold")
plt.ylabel("Membrane Potential (mV)")
plt.xlabel("Time (ms)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
