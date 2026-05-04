"""
Plot shape of crevasses for different wavenumbers and amplitudes
"""

import numpy as np
from importlib import reload


import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ion()
plt.isinteractive()

import sys
sys.path.append("../Code/")

import amp_equation_tools;reload(amp_equation_tools);import amp_equation_tools as amp
import stress_stream;reload(stress_stream);import stress_stream as stream

import plotAcrit;reload(plotAcrit);import plotAcrit


k = np.pi/2

Scrit,funcs,amp_eqn=amp.solveAmpEqn(k,ms=0.0,mb=0.0,n=3.0)

A = 0.5
x,surf1,bot1=stream.topo(k,A,funcs)
plt.plot(x,surf1,'k')
plt.plot(x,bot1,'k')

