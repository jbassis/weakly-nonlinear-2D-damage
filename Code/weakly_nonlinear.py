"""
Script to evaluate weakly non-linear solution
"""
from importlib import reload
import numpy as np
from scipy.signal import argrelextrema
import stress_stream
reload(stress_stream)
import stress_stream as stream
import non_hydro
reload(non_hydro)
import non_hydro as model

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ion()
plt.isinteractive()

def solveAmpEqn(k):
    # Determine critical stability parameter
    ms = 0.0
    mb = 0.0
    S0 = stream.Scrit(k,ms=ms,mb=mb)

    # Eigenvalues and eigenvector for leading order mode
    #eig,vec = stream.eigs(k,S0)

    # Create class for leading order term
    h11=stream.StreamFun(k,order=1,ms=ms,mb=mb)

    # Solve leading order term for coefficients
    h11.solve(S0,rhs=None)
    print("S0",S0,"1st order h11","s",h11.s,"b",h11.b)

    # Create a running dictionary of terms
    funcs={'h11':h11}


    # Solve for second order term psi22
    h22=stream.StreamFun(k,order=2,ms=ms,mb=mb)
    rhs22 = model.rhs22(k,S0,funcs)
    h22.solve(S0,rhs22)
    print("**********************2nd order h22","s",h22.s[0],"b",h22.b[0])

    funcs['h22']=h22

    # Second order mean field term, this term is hydrostatic
    rhs20=model.rhs20(k,S0,funcs)
    h20=stream.StreamFun(k,order=0,ms=ms,mb=mb)
    C,s,b=h20.solve(S0,rhs20)
    h20.stress=-4*S0*h20.h
    funcs['h20']=h20
    print("**********************2nd order h20","s",h20.s[0],"b",h20.b[0])


    # Third order term psi33
    h33=stream.StreamFun(k,order=3,ms=0.0,mb=mb)
    rhs33=model.rhs33(k,S0,funcs)
    h33.solve(S0,rhs33)
    funcs['h33']=h33
    print("**********************3rd order h33","s",h33.s[0],"b",h33.b[0])



    # Third order term psi31, needs to be treated differently
    rhs31=model.rhs31(k,S0,funcs)
    h31=stream.StreamFun(k,order=1,ms=0.0,mb=mb)
    S2=h31.solveS(S0,rhs31,funcs,S2=0.0)
    h31.set_amp_eqn(S0,rhs=rhs31,funcs=funcs,S2=None)
    h31.s,h31.b=0,0
    funcs['h31']=h31
    print("S2",S2)

    rhs42=model.rhs42(k,[S0,S2],funcs)
    h42=stream.StreamFun(k,order=2,ms=0.0,mb=mb)
    h42.solve(S0,rhs42)
    funcs['h42']=h42
    print("**********************2nd order h42","s",h42.s[0],"b",h42.b[0])

    rhs40=model.rhs40(k,[S0,S2],funcs)
    h40=stream.StreamFun(k,order=0,ms=0.0,mb=mb)
    h40.solve(S0,rhs40)
    h40.stress = -4*S0*h40.h - 4*S2*h20.h
    funcs['h40']=h40
    print("**********************2nd order h40","s",h40.s[0],"b",h40.b[0])



    rhs51=model.rhs51(k,[S0,S2],funcs)
    h51=stream.StreamFun(k,order=1,ms=0.0,mb=mb)
    S4=h51.solveS(S0,rhs51,funcs,S2=S2)
    print("S4",S4)
    funcs['h51']=h51


    # Fourth order term psi42


    # Mean field term psi40, this term is hydrostatic 
    amp_eqn= stream.AmplitudeEquation(S=[S0,S2,S4],h11=h11,rhs31=rhs31,rhs51=rhs51)
    return [S0,S2,S4],funcs,amp_eqn

# Set wavenumber
k = np.pi
Scrits,funcs,amp_eqn = solveAmpEqn(k)


# Fourth order term psi42


# Mean field term psi40, this term is hydrostatic 
#amp_eqn= stream.AmplitudeEquation(S=[S0,S2,S4],h11=h11,rhs31=rhs31,rhs51=rhs51)



S= np.linspace(-5,10,501)
x = np.linspace(0,4*2*np.pi/k,5001)

dAdt1=amp_eqn(S=S,A=0.01,order=5)
dAdt2=amp_eqn(S=S,A=-0.01,order=5)
dA =dAdt1-dAdt2
Acrit= amp_eqn.Acrit(S,order=5)
dam1 = np.array(stream.amp2dam(k,Acrit[1],funcs))
dam2 = np.array(stream.amp2dam(k,Acrit[2],funcs))

filter1 = ((Acrit[1]*k)<1.0) & (~np.isnan(Acrit[1]))
filter2 =  ((Acrit[2]*k)<1.0) & (~np.isnan(Acrit[2]))

plt.clf()
plt.plot(S[dA>0],Acrit[0][dA>0],color='dodgerblue',linewidth=3,linestyle=':')
plt.plot(S[dA<=0],Acrit[0][dA<=0],color='dodgerblue',linewidth=3,linestyle='-')
plt.plot(S,dam1,color='dodgerblue',linewidth=3,linestyle=':')
plt.plot(S,dam2,color='dodgerblue',linewidth=3,linestyle=':')
plt.plot(S[S>0],1/S[S>0],color='gray',linestyle='--',linewidth=3)
plt.plot(S[S>0],2/S[S>0],color='gray',linestyle='--',linewidth=3)

if k==np.pi/4:
    pi_over_2S = np.array([0.95, 1.05, 1.225, 1.65, 1.0, 1.0, 1.1, 1.3375])
    pi_over_2A = np.array([0.1, 0.2, 0.3, 0.35, 0.05, 0.15, 0.25, 0.325]) 
elif k==np.pi/2:
    pi_over_2S = np.array([0.715, 0.775, 1.05, 1.325, 1.15, 0.85, 0.745, 0.685])
    pi_over_2A = np.array([0.2, 0.3, 0.4, 0.5, 0.45, 0.35, 0.25, 0.15]) 
elif k==np.pi:
    pi_over_2S = np.array([1.55, 0.825, 0.225, 0.205, 0.23, 0.225, 0.2125, 1.3, 1.485, 1.225, 1.05, 0.95])
    pi_over_2A = np.array([0.3, 0.35, 0.25, 0.2, 0.15, 0.1, 0.275, 0.325, 0.31, 0.33, 0.34, 0.285])
elif k==2*np.pi:
    pi_over_2S = np.array([1.975,2.1,2.25,1.75, 0.375, 2.45, 2.95, 1.95, 1.2, 0.075])
    pi_over_2A = np.array([0.0975,0.095,0.1*0.93,0.1, 0.05, 0.09, 0.08, 0.07, 0.06, 0.04])
elif k==4*np.pi:
    pi_over_2S = np.array([4.75,4.0,3.65,3.0,2.75,2.0,1.85])
    pi_over_2A = np.array([0.01,0.0105,0.011,0.0115,0.012,0.01305,0.0135])
elif k==8*np.pi:
    pi_over_2S = np.array([0.1,1.0,2.0,3.0,4.0,7.5,6.1,4.85,4.0,2.95,2.25,2.0,1.85])
    pi_over_2A = np.array([0.0001,0.0001575,0.0001675,0.0001675,0.0001725,0.0002,0.000275,0.0003,0.000325,0.00035,0.000375,0.000385,0.000395])




pi_over_2D = stream.amp2dam(k,pi_over_2A,funcs)
plt.errorbar(pi_over_2S,pi_over_2D,yerr=0.025,xerr=0.2,fmt=' ',color='r',marker='o',markersize=5)
# Melt = 1.0 k = pi/4
amp_melt = [0.25,0.14,0.1]
S0_melt = [1.75,1.5,1.3]
# Melt = 1.0, k=pi/2
amp_melt = [0.085, 0.1, 0.15, 0.16,0.17]
S0_melt = [0.975, 1.15, 1.85,1.75, 1.95]
# Melt = 1.0, k=pi
amp_melt = [0.15,0.2,0.25]
S0_melt = [0.95,1.05,1.15]

# Melt = 1.0, k=8*pi
amp_melt = [0.095,0.1]
S0_melt = [4.0,3.95]

dam_melt = stream.amp2dam(k,amp_melt,funcs)
plt.plot(S0_melt,dam_melt,'bd')
plt.xlabel(r'$S_0$')
plt.ylabel(r"Damage")
plt.grid()
plt.ylim([-0.1,1.0])
plt.xlim([-5,8])



#k = 4*np.pi
#Scrits,funcs,amp_eqn = solveAmpEqn(k)
#pi_over_2S = np.array([4.75,4.0,3.65,3.0,2.75,2.0,1.85])
#pi_over_2A = np.array([0.01,0.0105,0.011,0.0115,0.012,0.01305,0.0135])
#data = stream.amp2dam(x,k,pi_over_2A,funcs)
#plt.errorbar(pi_over_2S,data,yerr=0.025,xerr=0.2,fmt=' ',color='b',marker='s',markersize=5)

plt.draw()







