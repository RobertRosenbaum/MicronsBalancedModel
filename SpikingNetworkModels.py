# This file contains implementations of recurrent spiking network models

import numpy as np
import matplotlib.pyplot as plt
from time import time as tm


# E-I recurrent EIF spiking network with current-based stimulus input
def EIFNetworkSimRecord(J,Jx,X,X0,Ne,NeuronParameters,tau,Nt,dt,maxns,dtRec):

  # N=Total number of neurons in network
  # Ni=number of inhibitory
  N=len(J)
  Ni=N-Ne

  # Get sub-network connectivity between cell types
  Jee=J[:Ne,:Ne]
  Jei=J[:Ne,Ne:]
  Jie=J[Ne:,:Ne]
  Jii=J[Ne:,Ne:]
  Jex=Jx[:Ne,:]
  Jix=Jx[Ne:,:]

  # EIF Neuron parameters
  Cm=NeuronParameters['Cm']
  gL=NeuronParameters['gL']
  EL=NeuronParameters['EL']
  Vth=NeuronParameters['Vth']
  Vre=NeuronParameters['Vre']
  Vlb=NeuronParameters['Vlb']
  DeltaT=NeuronParameters['DeltaT']
  VT=NeuronParameters['VT']

  # Synaptic time constants
  taue=tau[0]
  taui=tau[1]

  # Baseline (time-constant) external input
  X0e=X0[0]
  X0i=X0[1]

  # Initialize membrane potentials and synaptic currents
  Ve=np.random.rand(Ne)*(VT-Vre)+Vre
  Vi=np.random.rand(Ni)*(VT-Vre)+Vre
  Iee=np.zeros(Ne)
  Iei=np.zeros(Ne)
  Iie=np.zeros(Ni)
  Iii=np.zeros(Ni)

  NtRec=int(np.ceil(Nt*dt/dtRec))
  IeeRec = np.zeros((Ne,NtRec))
  IeiRec = np.zeros((Ne, NtRec))
  IieRec = np.zeros((Ni, NtRec))
  IiiRec = np.zeros((Ni, NtRec))
  IexRec = np.zeros((Ne, NtRec))
  IixRec = np.zeros((Ni, NtRec))
  VeRec = np.zeros((Ne, NtRec))
  ViRec = np.zeros((Ni, NtRec))

  # Number of exc and inh spikes so far is zero
  nespike=0
  nispike=0
  # Flag if there are too many spikes
  TooManySpikes=False
  # Initialize spike train arrays with -1's (interpreted as nulls)
  # and maximum number of allowed spikes
  se=-1.0+np.zeros((2,maxns))
  si=-1.0+np.zeros((2,maxns))
  # Time loop
  for i in range(Nt):


      # External inputs
      Iex=Jex@X[:,i]+X0e
      Iix=Jix@X[:,i]+X0i

      # Store recorded variables
      IeeRec[:, int(i * dt / dtRec)] += Iee
      IieRec[:, int(i * dt / dtRec)] += Iie
      IeiRec[:, int(i * dt / dtRec)] += Iei
      IiiRec[:, int(i * dt / dtRec)] += Iii
      IexRec[:, int(i * dt / dtRec)] += Iex
      IixRec[:, int(i * dt / dtRec)] += Iix
      VeRec[:, int(i * dt / dtRec)] += Ve
      ViRec[:, int(i * dt / dtRec)] += Vi

      # Euler update to V
      Ve=Ve+(dt/Cm)*(Iee+Iei+Iex+gL*(EL-Ve)+DeltaT*np.exp((Ve-VT)/DeltaT))
      Vi=Vi+(dt/Cm)*(Iie+Iii+Iix+gL*(EL-Vi)+DeltaT*np.exp((Vi-VT)/DeltaT))
      Ve=np.maximum(Ve,Vlb)
      Vi=np.maximum(Vi,Vlb)

      # Find which E neurons spiked
      Ispike = np.nonzero(Ve>=Vth)[0]
      if Ispike.any() and not(TooManySpikes):
          # Store spike times and neuron indices
          if nespike+len(Ispike)<=maxns:
              se[0,nespike:nespike+len(Ispike)]=dt*i
              se[1,nespike:nespike+len(Ispike)]=Ispike
          else:
              TooManySpikes=True

          # Reset e mem pot.
          Ve[Ispike]=Vre

          # Update exc synaptic currents
          Iee=Iee+Jee[:,Ispike].sum(axis = 1)/taue
          Iie=Iie+Jie[:,Ispike].sum(axis = 1)/taue

          # Update cumulative number of e spikes
          nespike=nespike+len(Ispike)

      # Find which I neurons spiked
      Ispike=np.nonzero(Vi>=Vth)[0]
      if Ispike.any() and not(TooManySpikes):
          # Store spike times and neuron indices
          if nispike+len(Ispike)<=maxns :
              si[0,nispike:nispike+len(Ispike)]=dt*i
              si[1,nispike:nispike+len(Ispike)]=Ispike
          else:
              TooManySpikes=True

          # Reset i mem pot.
          Vi[Ispike]=Vre

          # Update inh synaptic currents
          Iei=Iei+Jei[:,Ispike].sum(axis = 1)/taui
          Iii=Iii+Jii[:,Ispike].sum(axis = 1)/taui

          # Update cumulative number of i spikes
          nispike=nispike+len(Ispike)

      if TooManySpikes:
          print('Too many spikes. Exiting sim at time t =',i*dt)
          break

      # Euler update to synaptic currents
      Iee=Iee-dt*Iee/taue
      Iei=Iei-dt*Iei/taui
      Iie=Iie-dt*Iie/taue
      Iii=Iii-dt*Iii/taui

  # Scale recordings to make them into averages
  IeeRec *= (1 / int(dtRec / dt))
  IeiRec *= (1 / int(dtRec / dt))
  IieRec *= (1 / int(dtRec / dt))
  IiiRec *= (1 / int(dtRec / dt))
  IexRec *= (1 / int(dtRec / dt))
  IixRec *= (1 / int(dtRec / dt))
  VeRec *= (1 / int(dtRec / dt))
  ViRec *= (1 / int(dtRec / dt))

  Recording={};

  Recording['IeeRec'] = IeeRec
  Recording['IeiRec'] = IeiRec
  Recording['IieRec'] = IieRec
  Recording['IiiRec'] = IiiRec
  Recording['IexRec'] = IexRec
  Recording['IixRec'] = IixRec
  Recording['VeRec'] = VeRec
  Recording['ViRec'] = ViRec


  return se,si,Recording


# E-I recurrent EIF spiking network with current-based stimulus input
def EIFNetworkSim(J,Jx,X,X0,Ne,NeuronParameters,tau,Nt,dt,maxns,Ierecord):

  # N=Total number of neurons in network
  # Ni=number of inhibitory
  N=len(J)
  Ni=N-Ne

  # Get sub-network connectivity between cell types
  Jee=J[:Ne,:Ne]
  Jei=J[:Ne,Ne:]
  Jie=J[Ne:,:Ne]
  Jii=J[Ne:,Ne:]
  Jex=Jx[:Ne,:]
  Jix=Jx[Ne:,:]

  # EIF Neuron parameters
  Cm=NeuronParameters['Cm']
  gL=NeuronParameters['gL']
  EL=NeuronParameters['EL']
  Vth=NeuronParameters['Vth']
  Vre=NeuronParameters['Vre']
  Vlb=NeuronParameters['Vlb']
  DeltaT=NeuronParameters['DeltaT']
  VT=NeuronParameters['VT']

  # Synaptic time constants
  taue=tau[0]
  taui=tau[1]

  # Baseline (time-constant) external input
  X0e=X0[0]
  X0i=X0[1]

  # Initialize membrane potentials and synaptic currents
  Ve=np.random.rand(Ne)*(VT-Vre)+Vre
  Vi=np.random.rand(Ni)*(VT-Vre)+Vre
  Iee=np.zeros(Ne)
  Iei=np.zeros(Ne)
  Iie=np.zeros(Ni)
  Iii=np.zeros(Ni)

  # Number of neurons from which to record V. Initialize recorded V
  Nerecord=len(Ierecord)
  VeRec=np.zeros((Nt,Nerecord))

  # Number of exc and inh spikes so far is zero
  nespike=0
  nispike=0
  # Flag if there are too many spikes
  TooManySpikes=False
  # Initialize spike train arrays with -1's (interpreted as nulls)
  # and maximum number of allowed spikes
  se=-1.0+np.zeros((2,maxns))
  si=-1.0+np.zeros((2,maxns))
  # Time loop
  for i in range(Nt):

      # External inputs
      Iex=Jex@X[:,i]+X0e
      Iix=Jix@X[:,i]+X0i

      # Euler update to V
      Ve=Ve+(dt/Cm)*(Iee+Iei+Iex+gL*(EL-Ve)+DeltaT*np.exp((Ve-VT)/DeltaT))
      Vi=Vi+(dt/Cm)*(Iie+Iii+Iix+gL*(EL-Vi)+DeltaT*np.exp((Vi-VT)/DeltaT))
      Ve=np.maximum(Ve,Vlb)
      Vi=np.maximum(Vi,Vlb)

      # Find which E neurons spiked
      Ispike = np.nonzero(Ve>=Vth)[0]
      if Ispike.any() and not(TooManySpikes):
          # Store spike times and neuron indices
          if nespike+len(Ispike)<=maxns:
              se[0,nespike:nespike+len(Ispike)]=dt*i
              se[1,nespike:nespike+len(Ispike)]=Ispike
          else:
              TooManySpikes=True

          # Reset e mem pot.
          Ve[Ispike]=Vre

          # Update exc synaptic currents
          Iee=Iee+Jee[:,Ispike].sum(axis = 1)/taue
          Iie=Iie+Jie[:,Ispike].sum(axis = 1)/taue

          # Update cumulative number of e spikes
          nespike=nespike+len(Ispike)

      # Find which I neurons spiked
      Ispike=np.nonzero(Vi>=Vth)[0]
      if Ispike.any() and not(TooManySpikes):
          # Store spike times and neuron indices
          if nispike+len(Ispike)<=maxns :
              si[0,nispike:nispike+len(Ispike)]=dt*i
              si[1,nispike:nispike+len(Ispike)]=Ispike
          else:
              TooManySpikes=True

          # Reset i mem pot.
          Vi[Ispike]=Vre

          # Update inh synaptic currents
          Iei=Iei+Jei[:,Ispike].sum(axis = 1)/taui
          Iii=Iii+Jii[:,Ispike].sum(axis = 1)/taui

          # Update cumulative number of i spikes
          nispike=nispike+len(Ispike)

      if TooManySpikes:
          print('Too many spikes. Exiting sim at time t =',i*dt)
          break

      # Euler update to synaptic currents
      Iee=Iee-dt*Iee/taue
      Iei=Iei-dt*Iei/taui
      Iie=Iie-dt*Iie/taue
      Iii=Iii-dt*Iii/taui

      # Record V
      VeRec[i,:]=Ve[Ierecord]


  return se,si,VeRec


# E-I recurrent EIF spiking network with current-based stimulus input and membrane noise
def EIFNetworkSimWithNoise(J,Jx,X,X0,Ne,NeuronParameters,tau,Nt,dt,maxns,Ierecord,sigmav):
  N=len(J)
  Ni=N-Ne

  Jee=J[:Ne,:Ne]
  Jei=J[:Ne,Ne:]
  Jie=J[Ne:,:Ne]
  Jii=J[Ne:,Ne:]
  Jex=Jx[:Ne,:]
  Jix=Jx[Ne:,:]

  Cm=NeuronParameters['Cm']
  gL=NeuronParameters['gL']
  EL=NeuronParameters['EL']
  Vth=NeuronParameters['Vth']
  Vre=NeuronParameters['Vre']
  Vlb=NeuronParameters['Vlb']
  DeltaT=NeuronParameters['DeltaT']
  VT=NeuronParameters['VT']

  taue=tau[0]
  taui=tau[1]

  X0e=X0[0]
  X0i=X0[1]

  Ve=np.random.rand(Ne)*(VT-Vre)+Vre
  Vi=np.random.rand(Ni)*(VT-Vre)+Vre
  Iee=np.zeros(Ne)
  Iei=np.zeros(Ne)
  Iie=np.zeros(Ni)
  Iii=np.zeros(Ni)

  Nerecord=len(Ierecord)
  VeRec=np.zeros((Nt,Nerecord))

  # Noise parameters to achieve var(V)=sigmav for free mem pot.
  taum=Cm/gL
  noise_scale=sigmav*np.sqrt(2*dt/taum)

  nespike=0
  nispike=0
  TooManySpikes=False
  se=-1.0+np.zeros((2,maxns))
  si=-1.0+np.zeros((2,maxns))
  for i in range(Nt):
      # External inputs
      Iex=Jex@X[:,i]+X0e
      Iix=Jix@X[:,i]+X0i


      # Euler update to V
      Ve=Ve+(dt/Cm)*(Iee+Iei+Iex+gL*(EL-Ve)+DeltaT*np.exp((Ve-VT)/DeltaT))+noise_scale*np.random.randn(Ne)
      Vi=Vi+(dt/Cm)*(Iie+Iii+Iix+gL*(EL-Vi)+DeltaT*np.exp((Vi-VT)/DeltaT))+noise_scale*np.random.randn(Ni)
      Ve=np.maximum(Ve,Vlb)
      Vi=np.maximum(Vi,Vlb)

      # Find which E neurons spiked
      Ispike = np.nonzero(Ve>=Vth)[0]
      if Ispike.any() and not(TooManySpikes):
          # Store spike times and neuron indices
          if nespike+len(Ispike)<=maxns:
              se[0,nespike:nespike+len(Ispike)]=dt*i
              se[1,nespike:nespike+len(Ispike)]=Ispike
          else:
              TooManySpikes=True

          # Reset e mem pot.
          Ve[Ispike]=Vre

          # Update exc synaptic currents
          Iee=Iee+Jee[:,Ispike].sum(axis = 1)/taue
          Iie=Iie+Jie[:,Ispike].sum(axis = 1)/taue

          # Update cumulative number of e spikes
          nespike=nespike+len(Ispike)

      # Find which I neurons spiked
      Ispike=np.nonzero(Vi>=Vth)[0]
      if Ispike.any() and not(TooManySpikes):
          # Store spike times and neuron indices
          if nispike+len(Ispike)<=maxns :
              si[0,nispike:nispike+len(Ispike)]=dt*i
              si[1,nispike:nispike+len(Ispike)]=Ispike
          else:
              TooManySpikes=True

          # Reset i mem pot.
          Vi[Ispike]=Vre

          # Update inh synaptic currents
          Iei=Iei+Jei[:,Ispike].sum(axis = 1)/taui
          Iii=Iii+Jii[:,Ispike].sum(axis = 1)/taui

          # Update cumulative number of i spikes
          nispike=nispike+len(Ispike)

      if TooManySpikes:
          print('Too many spikes. Exiting sim at time t =',i*dt)
          break

      # Euler update to synaptic currents
      Iee=Iee-dt*Iee/taue
      Iei=Iei-dt*Iei/taui
      Iie=Iie-dt*Iie/taue
      Iii=Iii-dt*Iii/taui

      VeRec[i,:]=Ve[Ierecord]



  return se,si,VeRec


# E-I recurrent EIF spiking network with current-based stimulus input
def EIFNetworkSim1(J,Jx,X,X1,X0,Ne,NeuronParameters,tau,Nt,dt,maxns,Ierecord):
  N=len(J)
  Ni=N-Ne

  Jee=J[:Ne,:Ne]
  Jei=J[:Ne,Ne:]
  Jie=J[Ne:,:Ne]
  Jii=J[Ne:,Ne:]
  Jex=Jx[:Ne,:]
  Jix=Jx[Ne:,:]

  Cm=NeuronParameters['Cm']
  gL=NeuronParameters['gL']
  EL=NeuronParameters['EL']
  Vth=NeuronParameters['Vth']
  Vre=NeuronParameters['Vre']
  Vlb=NeuronParameters['Vlb']
  DeltaT=NeuronParameters['DeltaT']
  VT=NeuronParameters['VT']

  taue=tau[0]
  taui=tau[1]

  X0e=X0[0]
  X0i=X0[1]

  Ve=np.random.rand(Ne)*(VT-Vre)+Vre
  Vi=np.random.rand(Ni)*(VT-Vre)+Vre
  Iee=np.zeros(Ne)
  Iei=np.zeros(Ne)
  Iie=np.zeros(Ni)
  Iii=np.zeros(Ni)

  Nerecord=len(Ierecord)
  VeRec=np.zeros((Nt,Nerecord))

  nespike=0
  nispike=0
  TooManySpikes=False
  se=-1.0+np.zeros((2,maxns))
  si=-1.0+np.zeros((2,maxns))
  for i in range(Nt):
      # External inputs
      Iex=Jex@X[:,i]+X0e+X1[:Ne]
      Iix=Jix@X[:,i]+X0i+X1[Ne:]

      # Euler update to V
      Ve=Ve+(dt/Cm)*(Iee+Iei+Iex+gL*(EL-Ve)+DeltaT*np.exp((Ve-VT)/DeltaT))
      Vi=Vi+(dt/Cm)*(Iie+Iii+Iix+gL*(EL-Vi)+DeltaT*np.exp((Vi-VT)/DeltaT))
      Ve=np.maximum(Ve,Vlb)
      Vi=np.maximum(Vi,Vlb)

      # Find which E neurons spiked
      Ispike = np.nonzero(Ve>=Vth)[0]
      if Ispike.any() and not(TooManySpikes):
          # Store spike times and neuron indices
          if nespike+len(Ispike)<=maxns:
              se[0,nespike:nespike+len(Ispike)]=dt*i
              se[1,nespike:nespike+len(Ispike)]=Ispike
          else:
              TooManySpikes=True

          # Reset e mem pot.
          Ve[Ispike]=Vre

          # Update exc synaptic currents
          Iee=Iee+Jee[:,Ispike].sum(axis = 1)/taue
          Iie=Iie+Jie[:,Ispike].sum(axis = 1)/taue

          # Update cumulative number of e spikes
          nespike=nespike+len(Ispike)

      # Find which I neurons spiked
      Ispike=np.nonzero(Vi>=Vth)[0]
      if Ispike.any() and not(TooManySpikes):
          # Store spike times and neuron indices
          if nispike+len(Ispike)<=maxns :
              si[0,nispike:nispike+len(Ispike)]=dt*i
              si[1,nispike:nispike+len(Ispike)]=Ispike
          else:
              TooManySpikes=True

          # Reset i mem pot.
          Vi[Ispike]=Vre

          # Update inh synaptic currents
          Iei=Iei+Jei[:,Ispike].sum(axis = 1)/taui
          Iii=Iii+Jii[:,Ispike].sum(axis = 1)/taui

          # Update cumulative number of i spikes
          nispike=nispike+len(Ispike)

      if TooManySpikes:
          print('Too many spikes. Exiting sim at time t =',i*dt)
          break

      # Euler update to synaptic currents
      Iee=Iee-dt*Iee/taue
      Iei=Iei-dt*Iei/taui
      Iie=Iie-dt*Iie/taue
      Iii=Iii-dt*Iii/taui

      VeRec[i,:]=Ve[Ierecord]



  return se,si,VeRec


# E-I recurrent EIF spiking network with current-based stimulus input and spike based input
def EIFNetworkSimF(J,Sf,Jf,Jx,X,X0,Ne,NeuronParameters,tau,Nt,dt,maxns,Ierecord):
  N=len(J)
  Ni=N-Ne

  Jee=J[:Ne,:Ne]
  Jei=J[:Ne,Ne:]
  Jie=J[Ne:,:Ne]
  Jii=J[Ne:,Ne:]
  Jex=Jx[:Ne,:]
  Jix=Jx[Ne:,:]

  Jef=Jf[:Ne,:]
  Jif=Jf[Ne:,:]
    

  Cm=NeuronParameters['Cm']
  gL=NeuronParameters['gL']
  EL=NeuronParameters['EL']
  Vth=NeuronParameters['Vth']
  Vre=NeuronParameters['Vre']
  Vlb=NeuronParameters['Vlb']
  DeltaT=NeuronParameters['DeltaT']
  VT=NeuronParameters['VT']

  taue=tau[0]
  taui=tau[1]
  tauf=tau[2]

  X0e=X0[0]
  X0i=X0[1]

  Ve=np.random.rand(Ne)*(VT-Vre)+Vre
  Vi=np.random.rand(Ni)*(VT-Vre)+Vre
  Iee=np.zeros(Ne)
  Iei=np.zeros(Ne)
  Iie=np.zeros(Ni)
  Iii=np.zeros(Ni)
  Ief=np.zeros(Ne)
  Iif=np.zeros(Ni)

  Nerecord=len(Ierecord)
  VeRec=np.zeros((Nt,Nerecord))

  nespike=0
  nispike=0
  TooManySpikes=False
  se=-1.0+np.zeros((2,maxns))
  si=-1.0+np.zeros((2,maxns))
  for i in range(Nt):
      # External inputs
      Iex=Jex@X[:,i]+X0e
      Iix=Jix@X[:,i]+X0i
    
      Ief=Ief+dt*(-Ief+Jef@Sf[:,i])/tauf
      Iif=Iif+dt*(-Iif+Jif@Sf[:,i])/tauf

      # if i%5000==0:
      #   print(Ief.mean())

      # Euler update to V
      Ve=Ve+(dt/Cm)*(Iee+Iei+Iex+Ief+gL*(EL-Ve)+DeltaT*np.exp((Ve-VT)/DeltaT))
      Vi=Vi+(dt/Cm)*(Iie+Iii+Iix+Iif+gL*(EL-Vi)+DeltaT*np.exp((Vi-VT)/DeltaT))
      Ve=np.maximum(Ve,Vlb)
      Vi=np.maximum(Vi,Vlb)

      # Find which E neurons spiked
      Ispike = np.nonzero(Ve>=Vth)[0]
      if Ispike.any() and not(TooManySpikes):
          # Store spike times and neuron indices
          if nespike+len(Ispike)<=maxns:
              se[0,nespike:nespike+len(Ispike)]=dt*i
              se[1,nespike:nespike+len(Ispike)]=Ispike
          else:
              TooManySpikes=True

          # Reset e mem pot.
          Ve[Ispike]=Vre

          # Update exc synaptic currents
          Iee=Iee+Jee[:,Ispike].sum(axis = 1)/taue
          Iie=Iie+Jie[:,Ispike].sum(axis = 1)/taue

          # Update cumulative number of e spikes
          nespike=nespike+len(Ispike)

      # Find which I neurons spiked
      Ispike=np.nonzero(Vi>=Vth)[0]
      if Ispike.any() and not(TooManySpikes):
          # Store spike times and neuron indices
          if nispike+len(Ispike)<=maxns :
              si[0,nispike:nispike+len(Ispike)]=dt*i
              si[1,nispike:nispike+len(Ispike)]=Ispike
          else:
              TooManySpikes=True

          # Reset i mem pot.
          Vi[Ispike]=Vre

          # Update inh synaptic currents
          Iei=Iei+Jei[:,Ispike].sum(axis = 1)/taui
          Iii=Iii+Jii[:,Ispike].sum(axis = 1)/taui

          # Update cumulative number of i spikes
          nispike=nispike+len(Ispike)

      if TooManySpikes:
          print('Too many spikes. Exiting sim at time t =',i*dt)
          break

      # Euler update to synaptic currents
      Iee=Iee-dt*Iee/taue
      Iei=Iei-dt*Iei/taui
      Iie=Iie-dt*Iie/taue
      Iii=Iii-dt*Iii/taui

      VeRec[i,:]=Ve[Ierecord]

  print(Ief.mean())

  return se,si,VeRec



# E-I recurrent EIF spiking network with current-based stimulus input and Hebbian plasticity
def EIFNetworkSimHebb(J,Jx,X,X0,Ne,NeuronParameters,tau,Nt,dt,maxns,Ierecord,etaHebb=0.0,tauHebb=None):
  N=len(J)
  Ni=N-Ne

    
  if etaHebb>1e-9:
    HebbPlast=True
  else:
    HebbPlast=False
    
  Jee=J[:Ne,:Ne]
  Jei=J[:Ne,Ne:]
  Jie=J[Ne:,:Ne]
  Jii=J[Ne:,Ne:]
  Jex=Jx[:Ne,:]
  Jix=Jx[Ne:,:]
    
  EEmask=(Jee!=0)

  Cm=NeuronParameters['Cm']
  gL=NeuronParameters['gL']
  EL=NeuronParameters['EL']
  Vth=NeuronParameters['Vth']
  Vre=NeuronParameters['Vre']
  Vlb=NeuronParameters['Vlb']
  DeltaT=NeuronParameters['DeltaT']
  VT=NeuronParameters['VT']

  taue=tau[0]
  taui=tau[1]

  X0e=X0[0]
  X0i=X0[1]
    
  xeHebb=np.zeros(Ne)    

  Ve=np.random.rand(Ne)*(VT-Vre)+Vre
  Vi=np.random.rand(Ni)*(VT-Vre)+Vre
  Iee=np.zeros(Ne)
  Iei=np.zeros(Ne)
  Iie=np.zeros(Ni)
  Iii=np.zeros(Ni)

  Nerecord=len(Ierecord)
  VeRec=np.zeros((Nt,Nerecord))

  nespike=0
  nispike=0
  TooManySpikes=False
  se=-1.0+np.zeros((2,maxns))
  si=-1.0+np.zeros((2,maxns))
  for i in range(Nt):
      # External inputs
      Iex=Jex@X[:,i]+X0e
      Iix=Jix@X[:,i]+X0i

      # Euler update to V
      Ve=Ve+(dt/Cm)*(Iee+Iei+Iex+gL*(EL-Ve)+DeltaT*np.exp((Ve-VT)/DeltaT))
      Vi=Vi+(dt/Cm)*(Iie+Iii+Iix+gL*(EL-Vi)+DeltaT*np.exp((Vi-VT)/DeltaT))
      Ve=np.maximum(Ve,Vlb)
      Vi=np.maximum(Vi,Vlb)

      # Find which E neurons spiked
      Ispike = np.nonzero(Ve>=Vth)[0]
      if Ispike.any() and not(TooManySpikes):
          # Store spike times and neuron indices
          if nespike+len(Ispike)<=maxns:
              se[0,nespike:nespike+len(Ispike)]=dt*i
              se[1,nespike:nespike+len(Ispike)]=Ispike
          else:
              TooManySpikes=True

          # Reset e mem pot.
          Ve[Ispike]=Vre

          # Update exc synaptic currents
          Iee=Iee+Jee[:,Ispike].sum(axis = 1)/taue
          Iie=Iie+Jie[:,Ispike].sum(axis = 1)/taue

          # Update cumulative number of e spikes
          nespike=nespike+len(Ispike)
          
          if HebbPlast:
            #Jee[:,Ispike]=Jee[:,Ispike]+etaHebb*(xeHebb)*(Jee[:,Ispike]!=0)
            Jee[Ispike,:]=Jee[Ispike,:]+etaHebb*(Jee[Ispike,:]!=0)*(xeHebb)
            xeHebb[Ispike]=xeHebb[Ispike]+1/tauHebb


      # Find which I neurons spiked
      Ispike=np.nonzero(Vi>=Vth)[0]
      if Ispike.any() and not(TooManySpikes):
          # Store spike times and neuron indices
          if nispike+len(Ispike)<=maxns :
              si[0,nispike:nispike+len(Ispike)]=dt*i
              si[1,nispike:nispike+len(Ispike)]=Ispike
          else:
              TooManySpikes=True

          # Reset i mem pot.
          Vi[Ispike]=Vre

          # Update inh synaptic currents
          Iei=Iei+Jei[:,Ispike].sum(axis = 1)/taui
          Iii=Iii+Jii[:,Ispike].sum(axis = 1)/taui

          # Update cumulative number of i spikes
          nispike=nispike+len(Ispike)

      if TooManySpikes:
          print('Too many spikes. Exiting sim at time t =',i*dt)
          break

      # Euler update to synaptic currents
      Iee=Iee-dt*Iee/taue
      Iei=Iei-dt*Iei/taui
      Iie=Iie-dt*Iie/taue
      Iii=Iii-dt*Iii/taui
      xeHebb=xeHebb-dt*xeHebb/tauHebb

      VeRec[i,:]=Ve[Ierecord]

      #Jee=Jee+etaHebb*np.outer(xeHebb,xeHebb)*EEmask


  return se,si,VeRec
