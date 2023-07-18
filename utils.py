import numpy as np
import matplotlib.pyplot as plt
from time import time as tm


# Function to generate blockwise ER connection matrix
# NsPre = tuple of ints containing number of pre neurons in each block
# Jm = matrix connection weights in each block
# P = matrix of connection probs in each block
# NsPost = number of post neurons in each block
# If NsPost == None, connectivity is assumed recurrent (so NsPre=NsPost)
def GetBlockErdosRenyi(NsPre,Jm,P,NsPost=None):

  if NsPost==None:
    NsPost=NsPre

  # # If Jm is a 1D array, reshape it to column vector
  # if len(Jm.shape)==1:
  #   Jm = np.array([Jm]).T
  # if len(P.shape)==1:
  #   P = np.array([P]).T

  Npre = int(np.sum(NsPre))
  Npost = int(np.sum(NsPost))
  cNsPre = np.cumsum(np.insert(NsPre,0,0)).astype(int)
  cNsPost = np.cumsum(np.insert(NsPost,0,0)).astype(int)
  J = np.zeros((Npost,Npre))

  for j1,N1 in enumerate(NsPost):
    for j2,N2 in enumerate(NsPre):
      J[cNsPost[j1]:cNsPost[j1+1],cNsPre[j2]:cNsPre[j2+1]]=Jm[j1,j2]*(np.random.binomial(1, P[j1,j2], size=(N1, N2)))
  return J


# # Create a smooth Gaussian process
# def MakeSmoothGaussianProcess(taux,Nt,dt):
#   taus=np.arange(-3*taux,3*taux,dt)
#   K=np.exp(-taus**2/taux**2)
#   K=K/K.sum()
#   X=np.random.randn(Nt)/np.sqrt(dt)
#   X=np.convolve(K,X,'same')*dt
#   return X


# Create a smooth Gaussian process by convolving
# white noise with a Gaussian kernel.
# Noise will have variance=1
def MakeSmoothGaussianProcess(taux,Nt,dt):
  taus=np.arange(-3*taux,3*taux,dt)
  K=(1/(taux*np.sqrt(2*np.pi)))*np.exp(-taus**2/(2*taux**2))
  K=K/(dt*K.sum())
  X=np.random.randn(Nt)/np.sqrt(dt)
  X=np.sqrt(2*np.sqrt(np.pi)*taux)*np.convolve(K,X,'same')*dt
  return X


def PoissonProcess(r,dt,n=1,T=None,rep='sparse'):

  # If r is a 2D array, then there are multiple rates
  # and they are inhomogeneous in time.
  # r is interpreted as (neuron)x(time)
  # ie, r[j,k] is the rate of neuron j at time index k.
  # If rep=='full' then s has the same shape.
  if len(r.shape)==2:
    s = np.random.binomial(1,r*dt)/dt
    if rep == 'sparse':
      [I,J]=np.nonzero(s)
      temp=np.zeros((2,len(I)))
      temp[0,:]=J*dt
      temp[1,:]=I
      s=temp
      temp=np.argsort(s[0,:])
      s=s[:,temp]

  else:
    print('These input options are not yet implemented.')

  return s


# Returns 2D array of spike counts from sparse spike train, s.
# Counts spikes over window size winsize.
# h is represented as (neuron)x(time)
# so h[j,k] is the spike count of neuron j at time window k
def GetSpikeCounts(s,winsize,N,T):

  xedges=np.arange(0,N+1,1)
  yedges=np.arange(0,T+winsize,winsize)
  h,_,_=np.histogram2d(s[1,:],s[0,:],bins=[xedges,yedges])
  return h

