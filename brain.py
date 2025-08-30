#spiking cnn simulator, but not really
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
                        if neuron_type[idx] == 'I': W[idx,n_idx]*=-1

# STP & adaptation
U = 0.5
tau_rec = 100
syn_x = np.ones((N,N))
b = 0.1
w_adapt = np.zeros(N)

# Input currents
I = np.zeros((N,len(time)))
for t_idx, t in enumerate(time):
    I[:,t_idx] = 1.0 + 0.3*np.random.randn(N)
    I[(t_idx%N_side)::N_side,t_idx] += 1.5*np.exp(-0.05*t)
    I[:N_side*t_idx//10,t_idx] += 1.0*np.sin(0.1*t)

# Membrane potentials
V = np.ones(N)*V_rest
spikes = np.zeros((N,len(time)))
last_spike_time = np.ones(N)*-refractory_time

# Precompute spikes
for t_idx, t in enumerate(time):
    for n in range(N):
        if t - last_spike_time[n] < refractory_time: continue

        I_syn = np.sum(W[:,n] * spikes[:,t_idx-1] * syn_x[:,n]) if t_idx>0 else 0
        V[n] += dt * (-(V[n]-V_rest) + R*(I[n,t_idx]+I_syn) - w_adapt[n])/tau

        if V[n]>=V_th:
            if neuron_type[n]=='B' and np.random.rand()<0.6:
                spikes[n,t_idx:t_idx+3 if t_idx+3<len(time) else len(time)] = 1
            else:
                spikes[n,t_idx] = 1
            V[n]=V_reset
            last_spike_time[n]=t
            w_adapt[n]+=b
            if t_idx+1<len(time): syn_x[:,n] -= U*spikes[:,t_idx]
        syn_x[:,n]+=(1-syn_x[:,n])*dt/tau_rec
        w_adapt[n]-=w_adapt[n]*dt/200

# 2D neuron positions
x_base, y_base = np.meshgrid(np.arange(N_side), np.arange(N_side))
x_base = x_base.flatten()
y_base = y_base.flatten()

plt.close('all')
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(-1,N_side)
ax.set_ylim(-1,N_side)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('black')
ax.set_title("Readable Biologically Inspired Neural Network", color='white')

scat = ax.scatter(x_base, y_base, c='cyan', s=50, edgecolors='white', alpha=0.7)

lines = []
line_alpha = []
for i in range(N):
    for j in range(N):
        if W[i,j]!=0:
            line, = ax.plot([],[], color='magenta', alpha=0.0)
            lines.append((i,j,line))
            line_alpha.append(0.0)

def update(frame):
    x = x_base + 0.2*np.sin(0.1*frame + y_base)
    y = y_base + 0.2*np.cos(0.1*frame + x_base)

    sizes = 50 + 300*spikes[:,frame]
    colors = np.zeros((N,4))
    non_spike_mask = spikes[:,frame]==0
    spike_mask = spikes[:,frame]==1

    colors[non_spike_mask] = plt.cm.cool(np.clip(0.3+0.7*np.sin(0.05*frame + x_base[non_spike_mask] + y_base[non_spike_mask]),0,1))
    for n in np.where(spike_mask)[0]:
        if neuron_type[n]=='E': colors[n]=[1,0,0,1]
        elif neuron_type[n]=='I': colors[n]=[1,0.5,0,1]
        else: colors[n]=[1,1,0,1]

    scat.set_offsets(np.c_[x,y])
    scat.set_sizes(sizes)
    scat.set_facecolor(colors)

    # Update lines (only show recent spikes)
    for idx, (i,j,line) in enumerate(lines):
        line_alpha[idx]*=0.85  # decay
        if spikes[i,frame]==1:
            line_alpha[idx]=0.7
            line.set_data([x[i],x[j]],[y[i],y[j]])
        line.set_alpha(line_alpha[idx])

    return [scat]+[line for _,_,line in lines]

ani = FuncAnimation(fig, update, frames=len(time), interval=80, blit=False)  # slower interval
plt.show()
