"""
Evolve k=2*pi amplitude disturbance using amplitude equation
"""

import numpy as np
import pylab as plt




k = np.pi
dt = 0.0001*2
t = 0

# Define initial amplitude of the disturbance
n = 4.0
S = 2.0 
x = np.linspace(0,1,201)
H = 1.0
A = 0.05
# Define amplitude equation
def amp_eqn(A,S):
    t1 = (-0.2753187353 + (-3.312317888*S + 0.0517380608)*n)/n
    t2 = 48.96416334
    t3 = 1588.020131

    t1 =(-14.5314113 + (5.369025677*S + 18.49976824)*n)/n
    t2 = -1032.629786
    t3 = 63533.23315
    return t1*A+t2*A**3+t3*A**5

def topo(A,x):
    s11 = -0.9777384749*np.cos(2*np.pi*k*x)*A 
    b11 = 0.02226151492*np.cos(2*np.pi*k*x)*A 
    s22 = -0.002629448526*np.cos(2*2*np.pi*k*x)*A**2
    b22 = 0.001550238310*np.cos(2*2*np.pi*k*x)*A**2
    s33 = -0.002400521599*np.cos(3*2*np.pi*k*x)*A**3
    b33 = 0.0001628535122*np.cos(3*2*np.pi*k*x)*A**3
    s42 = -24.13258993*np.cos(2*2*np.pi*k*x)*A**4
    b42 = -0.02735739625*np.cos(2*2*np.pi*k*x)*A**4
    s = s11+s22+s33+s42 
    b = b11+b22+b33+b42
    return s,b 

for i in range(1000):
    s,b=topo(A,x)
    if np.any((b+H-s)<0.4):
        break 
    plt.clf()
    plt.fill_between(x,-s,-b+H,color='gray')
    plt.plot(x,-b+H,'k')
    plt.plot(x,-s,'k')
    plt.title(t)
    plt.ylim([-0.5,1.5])
    plt.draw()
    plt.show()
    plt.pause(0.1)
    dAdt=amp_eqn(A,S)
    A = A + dAdt*dt
    H = H -H*dt
    t= t +dt
    

