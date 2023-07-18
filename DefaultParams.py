import numpy as np

# Time duration and step size of sim in ms
T=20000
dt=.1
time=np.arange(0,T,dt)
Nt=len(time)
Tburn=500
Nburn=int(Tburn/dt)

# Number of E and I neurons in network
Ne = 4000
Ni = 1000
Ns=[Ne,Ni]
N = Ne+Ni


# Connection probabilities and weights between
# E and I neurons
P = np.array([[.1,.1],[.1,.1]])
Jm = np.array([[25.0,-150.0],[112.5,-250.0]])/np.sqrt(N)

# Synaptic time constants in ms
tau=np.array([8.0,4.0])


# Neuron parameters
NeuronParams = dict()
NeuronParams['Cm']=1.0
NeuronParams['gL']=1/15.0
NeuronParams['EL']=-72.0
NeuronParams['Vth']=0.0
NeuronParams['Vre']=-75.0
NeuronParams['Vlb']=-100.0
NeuronParams['DeltaT']=1.0
NeuronParams['VT']=-55.0


# Mean-field variables
Q=np.array([[Ne,Ni],[Ne,Ni]])/N
Wmf = Jm*P*Q*np.sqrt(N)
Xmf = np.array([.036,.027])

# Baseline external input to each neuron
X0 = Xmf*np.sqrt(N)

# Compute balanced rates for E and I neurons.
# This gives a rough approx to the actual rates
# if the network is approximately balanced.
rBal = -np.linalg.inv(Wmf)@Xmf

# Stimulus params
StimDim = 10
taustim = 100.0
sigmastim = X0.mean()/20.0

