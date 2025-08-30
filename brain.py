import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- PARAMETERS ---
T = 300             
dt = 1
time = np.arange(0, T+dt, dt)

N_side = 20
N = N_side**2

V_rest = -65
V_reset = -70
V_th = -50
R = 10
tau = 10

# Connectivity with neighbors
W = np.zeros((N,N))
for i in range(N_side):
    for j in range(N_side):
        idx = i*N_side + j
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                ni,nj = i+di,j+dj
                if 0<=ni<N_side and 0<=nj<N_side:
                    n_idx = ni*N_side + nj
                    if n_idx != idx:
                        W[idx,n_idx] = np.random.uniform(0.2,0.5)

# Input currents
I = np.zeros((N,len(time)))
for t_idx, t in enumerate(time):
    I[:,t_idx] = 1.5 + 0.5*np.random.randn(N)
    I[(t_idx%N_side)::N_side,t_idx] += 2*np.exp(-0.05*t)
    I[:N_side*t_idx//10,t_idx] += 1.5*np.sin(0.1*t)

# Membrane potentials
V = np.ones(N)*V_rest
spikes = np.zeros((N,len(time)))

for t_idx, t in enumerate(time):
    for n in range(N):
        I_syn = np.sum(W[:,n]*spikes[:,t_idx-1] if t_idx>0 else 0)
        dV = (-(V[n]-V_rest) + R*(I[n,t_idx]+I_syn))/tau
        V[n] += dV*dt
        if V[n]>=V_th:
            V[n] = 30
            spikes[n,t_idx] = 1
            V[n] = V_reset

# 2D neuron positions
x_base, y_base = np.meshgrid(np.arange(N_side), np.arange(N_side))
x_base = x_base.flatten()
y_base = y_base.flatten()

plt.close('all')

# Animation setup
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(-1,N_side)
ax.set_ylim(-1,N_side)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('black')
ax.set_title("Living Neural Wavefront Simulation", color='white')

scat = ax.scatter(x_base, y_base, c='cyan', s=50, edgecolors='white', alpha=0.7)

# Connection lines
lines = []
for i in range(N):
    for j in range(N):
        if W[i,j] != 0:
            line, = ax.plot([], [], color='magenta', alpha=0.02)
            lines.append((i,j,line))

def update(frame):
    x = x_base + 0.2*np.sin(0.1*frame + y_base)
    y = y_base + 0.2*np.cos(0.1*frame + x_base)
    
    sizes = 50 + 300*spikes[:,frame]
    
    # Colors: red for spikes, colormap for others
    colors = np.zeros((N,4))  # RGBA
    non_spike_mask = spikes[:,frame]==0
    spike_mask = spikes[:,frame]==1
    
    # Non-spiking neurons: use colormap
    colors[non_spike_mask] = plt.cm.cool(
        np.clip(0.3+0.7*np.sin(0.05*frame + x_base[non_spike_mask] + y_base[non_spike_mask]),0,1)
    )
    # Spiking neurons: red
    colors[spike_mask] = np.array([1,0,0,1])  # RGBA red
    
    scat.set_offsets(np.c_[x,y])
    scat.set_sizes(sizes)
    scat.set_facecolor(colors)
    
    # Connection lines
    for i,j,line in lines:
        if spikes[i,frame]==1:
            line.set_data([x[i],x[j]],[y[i],y[j]])
            line.set_alpha(0.7)
        else:
            line.set_alpha(max(0.02,line.get_alpha()*0.9))
    
    all_artists = [scat]+[line for _,_,line in lines]
    return all_artists

# Use blit=False for local scripts to avoid _resize_id error
ani = FuncAnimation(fig, update, frames=len(time), interval=50, blit=False)
plt.show()
