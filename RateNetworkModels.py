import numpy as np


# def EIFNetworkSim(J,Jx,X,X0,Ne,NeuronParameters,tau,Nt,dt,maxns,Ierecord):

def RateNetworkSim(W,Wx,X,X0,fIcurve,tau,Nt,dt):
    N = len(W)
    r=np.zeros((N,Nt))

    NtRec = int(np.ceil(Nt * dt / dtRec))
    rRec = np.zeros((N, NtRec))
    r=np.zeros(N)
    for i in range(Nt-1):
        r = r + dt*(-r+fIcurve(W@r+Wx@X[:,i]+X0))/tau



    return r
