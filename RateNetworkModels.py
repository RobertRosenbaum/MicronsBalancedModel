import numpy as np


#def EIFNetworkSim(J,Jx,X,X0,Ne,NeuronParameters,tau,Nt,dt,maxns,Ierecord):

def RateNetworkSim(W,Wx,X,X0,fIcurve,tau,Nt,dt):
    N = len(W)
    r=np.zeros((N,Nt))

    
    
    for i in range(Nt-1):
        r[:,i+1] = r[:,i] + dt*(-r[:,i]+fIcurve(W@r[:,i]+Wx@X[:,i]+X0))/tau

    return r
