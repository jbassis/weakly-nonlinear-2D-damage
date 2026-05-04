"""
Macro functions to solve for amplitude equation
"""

import numpy as np
from importlib import reload


#import matplotlib
#matplotlib.use("TkAgg")
#import matplotlib.pyplot as plt
#plt.ion()
#plt.isinteractive()

import sys
sys.path.append("../Code/")

import stress_stream
reload(stress_stream)
import stress_stream as stream
import non_hydro
reload(non_hydro)
import non_hydro as model


class AmpEqn(object):
    def __init__(self,k,S=1,melt=0,n=3,crit=False,E=1):
        if crit==True:
            S=stream.Scrit(k,melt,E=E)
        self.E = E
        self.k= k
        self.S0 = S
        self.melt=melt
        self.crit = crit

        #Scrit,funcs,amp_eqn=solveAmpEqn(k,n=n)
        #S=Scrit[0]
        # Create class for leading order term
        h11=stream.StreamFun(k,order=1,harmonic=1,mb=melt,E=E)

        # Solve leading order term for coefficients
        h11.solve(S,rhs=None)
        
        # Create a running dictionary of terms
        funcs={'h11':h11}

        # Solve for second order term psi22
        h22=stream.StreamFun(k,order=2,harmonic=2,mb=melt,E=E)
        rhs22 = model.rhs22(k,S,funcs,E=E)
        h22.solve(S,rhs22)
        funcs['h22']=h22

        rhs20=model.rhs20(k,S,funcs)
        h20=stream.MeanField(order=2,rhs=rhs20)
        funcs['h20']=h20
        #h20.s = 0
        #h20.b = 0 
        #h20.stress = 0.0

        h33=stream.StreamFun(k,order=3,harmonic=3,mb=melt,E=E)
        rhs33=model.rhs33(k,S,funcs,E=E)
        h33.solve(S,rhs33)
        funcs['h33']=h33

        # Third order term psi31, needs to be treated differently
        rhs31,rhs31_S2=model.rhs31(k,S,funcs,E=E)
        h31=stream.StreamFun(k,order=3,harmonic=1,mb=melt,E=E)
        #S2=h31.solveS(S,rhs31,rhs31_S2,funcs,S2=0.0,n=1)
        #S2 = 0.0
        #K,non_newt=h31.solve_landau_crit(S,rhs31,h11,n=n);h31.K=K;h31.s,h31.b=0,0
        h31.solve_landau_project(S,rhs31,h11,n=n)
        #h31.s=h31.s[0]
        #h31.b=h31.b[0]
        #print(h31.s,h31.b)
        #K = h31.K
        #print(h31.K,h31.s,h31.b)
        #31.solve(S,rhs31,K=K,h=[h11.s,h11.b])
        #h31.s=0
        #h31.b=0
        funcs['h31']=h31

        #self.K = K 
        K=h31.K

        self.dqdS = stream.deigsdS(self.k,self.S0,mb=self.melt,E=E)
        S2 = (-K/self.dqdS)[0]
        self.S2=S2

        S2 = 0
        
        rhs42=model.rhs42(k,[S,S2],funcs,K2=K)
        h42=stream.StreamFun(k,order=4,harmonic=2,mb=melt)
        h42.solve(S,rhs42)
        funcs['h42']=h42

        rhs44=model.rhs44(k,[S,S2],funcs,K2=K)
        h44=stream.StreamFun(k,order=4,harmonic=4,mb=melt)
        h44.solve(S,rhs44)
        funcs['h44']=h44

        rhs40=model.rhs40(k,[S,S2],funcs,K2=K)
        h40=stream.MeanField(order=4,rhs=rhs40)
        #h40.s=0 
        #h40.b=0 
        #h40.stress=0.0
        funcs['h40']=h40
        rhs51,rhs51_S2=model.rhs51(k,[S,S2],funcs,K2=K)
        h51=stream.StreamFun(k,order=5,harmonic=1,mb=melt)
        #K2,non_newt=h51.solve_landau_crit(S,rhs51,h11,n=n)
        
        #self.K2 = K2
        #print(K,K2)
        #h51.solve(S,rhs31,K=K2,h=[h11.s,h11.b])
        h51.solve_landau_project(S,rhs51,h11,n=n)
        h51.s=h51.s[0]
        h51.b=h51.b[0]
        #h51.s,h51.b=0,0
        #print(h51.K,K2,h51.s,h51.b)
        funcs['h51']=h51

        #print("none newtonian",non_newt)
        self.eig = h11.eig
        self.K2=h51.K
        self.K=h31.K
        self.non_newt = h51.sigma_non_newt
        self.K0 = h51.sigma_non_newt + h11.eig
        #self.non_newt = non_newt
        #self.K0 = h11.eig+non_newt
        #self.K=K
        #self.K2=K2

        self.dqdS = stream.deigsdS(self.k,self.S0,mb=self.melt,E=E)
        self.funcs=funcs
        #print('k',k,self.dqdS,-self.dqdS*self.S0,self.K,self.K2)
    
    def __call__(self,S,A,order):
         if self.crit==True:
            A1_term = (self.eig+self.dqdS*(S-self.S0))+self.non_newt
         else:
            A1_term = self.eig + self.non_newt
         if order == 1:
              return A1_term*A 
         if order == 3:
              return A1_term*A+self.K*A**3
         elif order==5: 
              return A1_term*A+self.K*A**3+self.K2*A**5
         else:
              print("not an option")
         
    def Acrit(self,S,order):
         if order==5:
            alpha = (self.eig+self.dqdS*(S-self.S0))+self.non_newt
            beta = self.K
            gamma = self.K2
            A1 = np.zeros(np.shape(S))
            A2 = np.abs(np.sqrt(2)*np.sqrt(gamma*(-beta + np.sqrt(-4*alpha*gamma + beta**2)))/(2*gamma))
            A3 = np.abs(np.sqrt(-2*gamma*(beta + np.sqrt(-4*alpha*gamma + beta**2)))/(2*gamma))
         elif order==3:
            t1 =  (self.eig+self.dqdS*(S-self.S0))+self.non_newt
            t3 = self.K
            alpha = t1/t3
            A1 = np.zeros(np.shape(S))
            A3 = np.zeros(np.shape(S))
            A2 = np.sqrt(-alpha)
         return [A1,A2,A3]
    


         

def Acrit(k,S,melt,n=3,order=3):
     A1 = np.zeros(np.shape(S))
     A2 = np.zeros(np.shape(S))
     A3 = np.zeros(np.shape(S))
     for i in range(len(S)):
        amp_eqn=AmpEqn(k,S[i],melt=melt,n=n,crit=False)
        if order==5:
            alpha = amp_eqn.K0
            beta =  amp_eqn.K
            gamma = amp_eqn.K2
            A2[i] = np.abs(np.sqrt(2)*np.sqrt(gamma*(-beta + np.sqrt(-4*alpha*gamma + beta**2)))/(2*gamma))
            A3[i] = np.abs(np.sqrt(-2*gamma*(beta + np.sqrt(-4*alpha*gamma + beta**2)))/(2*gamma))
        else:
            #alpha = amp_eqn.K0/amp_eqn.K2
            #A2[i] = np.sqrt(-alpha)
            alpha = amp_eqn.K0
            beta = amp_eqn.K
            A2[i]=np.sqrt(-beta*alpha)/beta
            
     return A1,A2,A3

         


def solveAmpEqn(k,ms=0,mb=0.0,n=1,order=5,kin=1,E=1):
    # Determine critical stability parameter
    S0 = stream.Scrit(k,mb=mb,E=E)
        
    #Sc = stream.Scrit(k,ms=0.0,mb=0.0,E=1,kin=0)
    #print(S0,Sc)
    
    
    #S0=

    # Eigenvalues and eigenvector for leading order mode
    #eig,vec = stream.eigs(k,S0)

    # Create class for leading order term
    h11=stream.StreamFun(k,order=1,harmonic=1,ms=ms,mb=mb,E=E)

    # Solve leading order term for coefficients
    h11.solve(S0,rhs=None,kin=kin)
    
    # Create a running dictionary of terms
    funcs={'h11':h11}

    # Solve for second order term psi22
    h22=stream.StreamFun(k,order=2,harmonic=2,ms=ms,mb=mb,E=E)
    rhs22 = model.rhs22(k,S0,funcs,E=E)
    h22.solve(S0,rhs22,kin=kin)

    funcs['h22']=h22

    # Second order mean field term, this term is hydrostatic
    #h20=stream.StreamFun(k,order=2,harmonic=0,ms=ms,mb=mb)
    #rhs20=model.rhs20(k,S0,funcs)
    #h20.solve(S0,rhs20,kin=kin)
    #h20.stress=-4*S0*h20.h
    #h20.s=h20.s[0]
    #h20.b=h20.b[0]
    #h20.s = 0 
    #h20.b = 0
    #h20.stress = 0.0
    #h20.stress=-4*S0*h20.h
    
    rhs20=model.rhs20(k,S0,funcs)
    h20=stream.MeanField(order=2,rhs=rhs20)
    funcs['h20']=h20


    # Third order term psi33
    h33=stream.StreamFun(k,order=3,harmonic=3,ms=ms,mb=mb,E=E)
    rhs33=model.rhs33(k,S0,funcs,E=E)
    h33.solve(S0,rhs33,kin=kin)
    funcs['h33']=h33


    # Third order term psi31, needs to be treated differently
    rhs31,rhs31_S2=model.rhs31(k,S0,funcs,E=E)
    h31=stream.StreamFun(k,order=3,harmonic=1,ms=ms,mb=mb,E=E)
    S2=h31.solveS(S0,rhs31,rhs31_S2,funcs,S2=0.0,n=1)
    h31.s=h31.s[0]
    h31.b=h31.b[0]
    #h31.s,h31.b=0,0
    funcs['h31']=h31

    #h20=stream.StreamFun(k,order=2,harmonic=0,ms=ms,mb=mb)
    #rhs20=model.rhs20(k,S0,funcs)
    ##h20.solve(S0,rhs20,kin=kin)
    #h = S2/S0
    #h20.stress=-4*S0*h
    #h20.h = h
    #h20.s = -4*h/stream.d_s
    #h20.b = -4*h/stream.d_b
    #funcs['h20']=h20

    # Solve for Landau coefficient
    #K=h31.solve_landau_crit(S0,rhs31,funcs,n=n)
    #K=h11.solve_landau_crit(S0,rhs31,funcs,n=n)
    

    rhs42=model.rhs42(k,[S0,S2],funcs,E=E)
    h42=stream.StreamFun(k,order=4,harmonic=2,ms=ms,mb=mb,E=E)
    h42.solve(S0,rhs42,kin=kin)
    funcs['h42']=h42


    rhs44=model.rhs44(k,[S0,S2],funcs,E=E)
    h44=stream.StreamFun(k,order=4,harmonic=4,ms=ms,mb=mb,E=E)
    h44.solve(S0,rhs44,kin=kin)
    funcs['h44']=h44

    rhs40=model.rhs40(k,[S0,S2],funcs)
    h40=stream.MeanField(order=4,rhs=rhs40)
    funcs['h40']=h40


    rhs51,rhs51_S4=model.rhs51(k,[S0,S2],funcs,E=E)
    #part_sol = stream.ParticularSol(h11)
    #stress_bc = part_sol.stress_rhs(n=3)
    #kin_bc = part_sol.kinematic_rhs(n=3)
    #rhs51_non_newt = part_sol.non_newtonian_rhs(n=1e6)
    #print(rhs51)
    #rhs51 = rhs51 + rhs51_non_newt
    #print(rhs51)
    

    h51=stream.StreamFun(k,order=5,harmonic=1,ms=ms,mb=mb,E=E)
    S4=h51.solveS(S0,rhs51,rhs51_S4,funcs,S2=S2,n=n)
    #h51.s=0
    #h51.b=0
    funcs['h51']=h51


    # Second order mean field term, this term is hydrostatic
    #h40=stream.StreamFun(k,order=4,harmonic=0,ms=ms,mb=mb)
    #rhs40=model.rhs40(k,[S0,S2],funcs)
    #h = S4/S0
    #h40.stress=-4*S2*h -4*S4*h20.h
    #h40.h = h
    #h40.s = -4*h/stream.d_s
    #h40.b = -4*h/stream.d_b

    #print(h11.C)
    #print("S4",S4)
    #funcs['h40']=h40


    # Fourth order term psi42

   
    # Mean field term psi40, this term is hydrostatic 
  
    amp_eqn= stream.AmplitudeEquation(S=[S0,S2,S4],h11=h11,rhs31=rhs31,rhs51=rhs51,rhs31S2=rhs31_S2,rhs51S2=rhs51_S4,n=n,ms=ms,mb=mb,E=E)
    return [S0,S2,S4],funcs,amp_eqn

def evolve(k,S,A,funcs0,dt,melt_rate,order=5,Nt=1,n=3,crit=True,E=1):
        damage = []
        tt = []
        kk = []
        amp = []
        t = 0
        [S0,S2,S4],funcs0,amp_eqn=solveAmpEqn(k,n=n,E=E)
        for i in range(Nt):
            dam = stream.amp2dam(k,[A],funcs0,order=order)[0]
            #print(t,k,dam)
            tt.append(t)           
            damage.append(dam)
            kk.append(k)
            [S0,S2,S4],funcs,amp_eqn=solveAmpEqn(k,n=n,E=E)
            dAdt = amp_eqn(S,A,order=order)
            #print("k",k,"dAdt",dAdt)
            A = A + (dAdt)*dt 
            #k = k - 2*dt*k
            k = k*np.exp(-dt*(2.0*E+melt_rate))
            #k = k*np.exp(-dt*(1.0+melt_rate))
           
            #if k<np.pi/8:
            #     k = np.pi/4
                 
            t = t + dt
            dam = stream.amp2dam(k,[A],funcs0,order=order)[0]
            #A = stream.dam2amp(k,[dam],funcs0,order=order)
            #print(t,k,dam)
            if np.isnan(dam)==1:
                 break
            if dam>=1:
                 break
            if dam<1e-3:
                 break
            
            #amp.append(A)
        return kk,tt,damage#,amp

def evolveK(k,S,A,dt,melt_rate,n=3,order=5,Nt=1,crit=True,E=1,evolve_k = True):
        damage = []
        tt = []
        kk = []
        amp = []
        t = 0
        
        #amp_eqn=AmpEqn(k,S,melt=0,n=1,order=3)
        #amp_eqn0=AmpEqn(k,S,melt=0,n=3,order=3)
        #funcs0=amp_eqn0.funcs
        #Scrit,funcs0,amp_eqn=solveAmpEqn(k)
        #amp_eqn0=AmpEqn(k,Scrit[0],melt=melt_rate,n=n,order=order)
        #funcs0 = amp_eqn0.funcs
        #[S0,S2,S4],funcs0,amp_eqn=solveAmpEqn(k)
        #amp_eqn0=AmpEqn(k,S=S,melt=melt_rate,n=n,crit=crit)
        #funcs0 = amp_eqn0.funcs
        #print(funcs0['h11'].s,funcs0['h22'].s,funcs0['h20'].s,funcs0['h22'].s,funcs0['h31'].s,funcs0['h33'].s,funcs0['h40'].s,funcs0['h42'].s,funcs0['h44'].s)
        amp_eqn0=AmpEqn(k,S,melt_rate*0,n=n,crit=True,E=E)
        funcs0 = amp_eqn0.funcs
        for i in range(Nt):
            amp_eqn=AmpEqn(k,S,melt_rate,n=n,crit=crit,E=E)
            funcs=amp_eqn.funcs
            dam = stream.amp2dam(k,[A],funcs0,order=order)[0]
            #dam = stream.amp2dam(k,[A],amp_eqn.funcs,order=order)[0]
            #print(t,k,dam)
            tt.append(t)           
            damage.append(dam)
            kk.append(k)
            #amp_eqn=AmpEqn(k,S,melt=0,n=3,order=order)
            
            dAdt = amp_eqn(S,A,order=order).item()
            #dAdt = (n*(1-S)+melt_rate)*A

            #dAdt2 = amp_eqn0(S,A)
            #dAdt = 0.5*(dAdt+dAdt2)
            A = A + (dAdt)*dt 
            if evolve_k ==True:
               k = k*np.exp(-dt*(2*E+melt_rate))
               k = k*np.exp(-dt*(E))
               #k = k*np.exp(-dt*(2*E))
               #k = k
            #k = k*np.exp(-dt*(melt_rate))
            #print('t',t,'k',k,'dAdt',dAdt)
                 
            t = t + dt
            #dam = stream.amp2dam(k,[A],funcs,order=order)[0]
            #A = stream.dam2amp(k,[dam],funcs0,order=order)
            
            #if np.isnan(dam)==1:
            #     break
            if dam>=1:
                 break
            #if dam<1e-3:
            #     break
            
            #amp.append(A)
        return kk,tt,damage


def Acrit_solve(k,Slist,order,melt=0,n=1):
     from scipy import optimize
     Acrit = []
     def dadt(A,S):
          amp_eqn = AmpEqn(k,S=S,melt=melt,n=n,crit=False)
          return amp_eqn(S,A,order=order)
     for S in Slist:
          try:
               Ac = optimize.brentq(dadt, 1e-6, 1,args=(S))
          except:
              Ac = np.nan
          Acrit.append(Ac)
     return Acrit
     
     