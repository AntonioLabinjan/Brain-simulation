#~ simulation of spiking cnn, but not really
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
refractory_time = 5  # ms

# Neuron types: excitatory (E), inhibitory (I), bursting (B)
np.random.seed(42)
neuron_type = np.random.choice(['E','I','B'], size=N, p=[0.7,0.2,0.1])

# Connectivity
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
                        if neuron_type[idx] == 'I':
                            W[idx,n_idx] *= -1

# Short-term synaptic plasticity parameters
U = 0.5  # utilization
tau_rec = 100  # recovery time
syn_x = np.ones((N,N))  # available synaptic resources

# Adaptation parameters
a = 0.01
b = 0.1
w_adapt = np.zeros(N)  # adaptation current

# Input currents with background noise and wavefront
I = np.zeros((N,len(time)))
for t_idx, t in enumerate(time):
    I[:,t_idx] = 1.0 + 0.3*np.random.randn(N)
    I[(t_idx%N_side)::N_side,t_idx] += 1.5*np.exp(-0.05*t)
    I[:N_side*t_idx//10,t_idx] += 1.0*np.sin(0.1*t)

# Membrane potentials
V = np.ones(N)*V_rest
spikes = np.zeros((N,len(time)))
last_spike_time = np.ones(N)*-refractory_time

# Precompute spikes with STP, adaptation, bursting
for t_idx, t in enumerate(time):
    for n in range(N):
        if t - last_spike_time[n] < refractory_time:
            continue
        
        # Synaptic input with STP
        if t_idx>0:
            I_syn = np.sum(W[:,n] * spikes[:,t_idx-1] * syn_x[:,n])
        else:
            I_syn = 0
        
        dV = (-(V[n]-V_rest) + R*(I[n,t_idx]+I_syn) - w_adapt[n]) / tau
        V[n] += dV*dt
        
        # Check for spike
        if V[n] >= V_th:
            # Bursting neurons
            if neuron_type[n]=='B' and np.random.rand()<0.6:
                spikes[n,t_idx:t_idx+3 if t_idx+3<len(time) else len(time)] = 1
            else:
                spikes[n,t_idx] = 1
            
            V[n] = V_reset
            last_spike_time[n] = t
            
            # Update adaptation
            w_adapt[n] += b
            
            # Update STP
            if t_idx+1 < len(time):
                syn_x[:,n] = syn_x[:,n] - U*spikes[:,t_idx]
        
        # Recover STP
        syn_x[:,n] += (1 - syn_x[:,n]) * dt / tau_rec
        w_adapt[n] -= w_adapt[n]*dt/200  # slow adaptation decay

# 2D neuron positions
x_base, y_base = np.meshgrid(np.arange(N_side), np.arange(N_side))
x_base = x_base.flatten()
y_base = y_base.flatten()

plt.close('all')

# Animation
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(-1,N_side)
ax.set_ylim(-1,N_side)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('black')
ax.set_title("Advanced Biologically Inspired Neural Network", color='white')

scat = ax.scatter(x_base, y_base, c='cyan', s=50, edgecolors='white', alpha=0.7)

lines = []
for i in range(N):
    for j in range(N):
        if W[i,j]!=0:
            line, = ax.plot([],[], color='magenta', alpha=0.02)
            lines.append((i,j,line))

def update(frame):
    x = x_base + 0.2*np.sin(0.1*frame + y_base)
    y = y_base + 0.2*np.cos(0.1*frame + x_base)
    
    sizes = 50 + 300*spikes[:,frame]
    colors = np.zeros((N,4))
    non_spike_mask = spikes[:,frame]==0
    spike_mask = spikes[:,frame]==1
    
    # Non-spiking neurons
    colors[non_spike_mask] = plt.cm.cool(
        np.clip(0.3+0.7*np.sin(0.05*frame + x_base[non_spike_mask] + y_base[non_spike_mask]),0,1)
    )
    # Spiking neurons: E=red, I=orange, B=yellow
    for n in np.where(spike_mask)[0]:
        if neuron_type[n]=='E':
            colors[n]=[1,0,0,1]
        elif neuron_type[n]=='I':
            colors[n]=[1,0.5,0,1]
        else:
            colors[n]=[1,1,0,1]
    
    scat.set_offsets(np.c_[x,y])
    scat.set_sizes(sizes)
    scat.set_facecolor(colors)
    
    for i,j,line in lines:
        if spikes[i,frame]==1:
            line.set_data([x[i],x[j]],[y[i],y[j]])
            line.set_alpha(0.7)
        else:
            line.set_alpha(max(0.02,line.get_alpha()*0.9))
    
    all_artists = [scat]+[line for _,_,line in lines]
    return all_artists

ani = FuncAnimation(fig, update, frames=len(time), interval=50, blit=False)
plt.show()
