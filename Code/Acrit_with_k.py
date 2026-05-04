"""
Critical amplitudes based on amplitude equation and evolution of k
"""

import numpy as np
from importlib import reload


#import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ion()
#plt.isinteractive()

import sys
sys.path.append("../Code/")
import stress_stream
reload(stress_stream)
import stress_stream as stream
import amp_equation_tools;reload(amp_equation_tools);import amp_equation_tools as amp_tools

# Root finding
from scipy.optimize import brentq

# Specify wavenumber
k =np.pi

#
n=3

# Melt
melt=0

# Stability number
S0=0.3

# Amplitude?
amp = 0.05

evolve_k = True

order = 3

crit = False

kk1,tt1,dam1=amp_tools.evolveK(k,S=S0,A=amp,melt_rate=melt,dt=0.005,Nt=150,n=n,crit=crit,order=order,evolve_k=evolve_k)
kk2,tt2,dam2=amp_tools.evolveK(k,S=S0,A=amp,melt_rate=melt,dt=0.005,Nt=150,n=n,crit=crit,order=order,evolve_k=True)

def dAdt(S0,amp,melt,evolve_k):
    kk,tt,dam=amp_tools.evolveK(k,S=S0,A=amp,melt_rate=melt,dt=0.02,Nt=150,n=n,crit=crit,order=order,evolve_k=evolve_k)
    coeff = np.polyfit(tt, dam, 1)
    return coeff[0]

amp_list = np.arange(0.05,0.4,0.05) 

amp_eqn = amp_tools.AmpEqn(k,melt=melt,n=n,crit=True)
funcs=amp_eqn.funcs
dam1 = np.array(stream.amp2dam(k,amp_list,funcs,order=order))

dam = stream.amp2dam(k,amp_list,funcs,order=order)

#Scrit = []
#for amp in amp_list:
#    root = brentq(dAdt, 1e-3, 5, args=(amp,melt,evolve_k))
#    print('root',root)
#    Scrit.append(root)


#plt.clf()
#plt.plot(Scrit,dam1,'ok')



Scrit2 = []
Slist = np.arange(1e-5,2,0.05)
evolve_k = True
for amp in amp_list:
    slope_list = []
    for S in Slist:
        kk,tt,dam=amp_tools.evolveK(k,S=S,A=amp,melt_rate=melt,dt=0.01,Nt=200,n=n,crit=crit,order=order,evolve_k=evolve_k)
        coeff = np.polyfit(tt, dam, 1)
        #slope_list.append(coeff[0])
        slope_list.append(coeff[0])
        #print('amp',amp,'S',S,'slope',slope_list[-1])
        if slope_list[-1]<0:
            break
    from numpy.polynomial import Polynomial
    from scipy.interpolate import CubicSpline
    p = Polynomial.fit(slope_list,Slist[0:len(slope_list)], deg=2)
    #p = CubicSpline(slope_list[::-1],Slist[0:len(slope_list)][::-1], bc_type='natural')
    Scrit2.append(p(0))
    print('Scrit',Scrit2[-1])
    

    #root = brentq(dAdt, 1e-3, 5, args=(amp,melt,evolve_k))
    #print('root',root)
    #Scrit2.append(root)

plt.clf()
plt.plot(Scrit2,dam1,'dr')



#kk1,tt1,dam1=amp_tools.evolveK(k,S=0.81,A=0.35,melt_rate=melt,dt=0.005,Nt=150,n=n,crit=crit,order=order,evolve_k=evolve_k)




# Notes
#crit_flag = True in initial condition for model AND evolve_k=True for amplitude equation
#   - crit_flag=false (n=3) underestimates for until amp = 0.2 and 0.25 and then it overestimates the crit solution

# crit_flag = False in initial condition for model AND evolve_k=True for amplitude equation
# Underestimates for t>0.8, except for amp=0.25, where it remains accurate up to t=1.0

# crit_flag = False in initial condition for model AND evolve_k=FALSE for amplitude equation
# Overestimates amp=0.1 and amp=0.15 after t~0.2 or t~0.25
# Overestimates amp=0.2 after t~0.5
# Underestimates amp=0.25 after t~0.2

# crit_flag = True in initial condition for model AND evolve_k=FALSE for amplitude equation
# amp=0.1 slightly overestimates after t~0.2
# amp=0.15 starts at red curve and then finds green curve and is accurate up to at least t~1.0
# amp=0.2 underestimates after about t~0.7 
# amp=0.25 numerical solution underestimated by amplitude equation after t~0.1


# Best fitting is crit_flag = False, evolve_k=True
# Tested with k=np.pi, amp = 0.25,melt = 0,1,2,3