"""
Estimate stress enhancement associated with crevasse placement to compare with Roger Buck's EPSL
force balance
"""

from numpy import *
import pylab as mlab
from importlib import reload



#Setup basic simulation parameters
rho_s = 0.0     # Density of snow
rho_m = 1020.0  # Density of ocean water
rho_ice = 920.0 # Density of ice
rho_w = 1020.0
sigma = rho_ice*(1-rho_ice/rho_w)/8*2

# Define dimensionless numbers that control evolution
S0 = (rho_s-rho_ice)/(sigma)  
S1 = (rho_w - rho_ice)/(sigma)
S=-0.25*S0*S1/(S1-S0)
print("Stability Number",S)
T11 = (rho_w-rho_s)/sigma/2
T22 = (2*rho_ice-rho_m-rho_s)/sigma/2
R = T22/T11

# Growth rates for newtonian solution
T=[T11,T22]
def growth_rates(k,sigma,T):
    beta = emath.sqrt(k**2+sigma)
    qAA = 2 * sigma * cosh(k / 2) * cosh(beta / 2) / (-2 * (k ** 2 + sigma / 2) * k * sinh(k / 2) * (2 + sigma) * cosh(beta / 2) + 2 * sinh(beta / 2) * ((2 + sigma) * k ** 2 - sigma) * beta * cosh(k / 2)) * T[0] + (-4 * k ** 3 * sinh(k / 2) * (2 + sigma) * cosh(beta / 2) + 4 * sinh(beta / 2) * ((2 + sigma) * k ** 2 - sigma) * beta * cosh(k / 2)) / (-2 * (k ** 2 + sigma / 2) * k * sinh(k / 2) * (2 + sigma) * cosh(beta / 2) + 2 * sinh(beta / 2) * ((2 + sigma) * k ** 2 - sigma) * beta * cosh(k / 2))
    qAB = 2 * T[1] * cosh(beta / 2) * cosh(k / 2) * sigma / (-cosh(beta / 2) * k * (2 + sigma) * (2 * k ** 2 + sigma) * sinh(k / 2) + 2 * sinh(beta / 2) * ((2 + sigma) * k ** 2 - sigma) * beta * cosh(k / 2))
    qBA = -2 * sinh(k / 2) * sinh(beta / 2) * sigma * T[1] / (sinh(beta / 2) * k * (2 + sigma) * (2 * k ** 2 + sigma) * cosh(k / 2) - 2 * ((2 + sigma) * k ** 2 - sigma) * beta * cosh(beta / 2) * sinh(k / 2))
    qBB = -2 * sigma * sinh(k / 2) * sinh(beta / 2) / (2 * cosh(k / 2) * (k ** 2 + sigma / 2) * k * (2 + sigma) * sinh(beta / 2) - 2 * ((2 + sigma) * k ** 2 - sigma) * beta * cosh(beta / 2) * sinh(k / 2)) * T[0] + (4 * k ** 3 * cosh(k / 2) * (2 + sigma) * sinh(beta / 2) - 4 * ((2 + sigma) * k ** 2 - sigma) * beta * cosh(beta / 2) * sinh(k / 2)) / (2 * cosh(k / 2) * (k ** 2 + sigma / 2) * k * (2 + sigma) * sinh(beta / 2) - 2 * ((2 + sigma) * k ** 2 - sigma) * beta * cosh(beta / 2) * sinh(k / 2))
    return qAA,qAB,qBA,qBB

def eigen(k,sigma,T):
    # Calculate growth rates based on velocities
    qAA,qAB,qBA,qBB=growth_rates(k,sigma,T)

    # Determinant as a function growth rate sigma and k
    deter = (qAA-sigma)*(qBB-sigma)-qAB*qBA

    

    return deter

def find_roots(sigma):
    """

    """



