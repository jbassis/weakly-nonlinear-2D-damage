from importlib import reload
import numpy as np
import stress_stream
reload(stress_stream)
import stress_stream as stream
import non_hydro
reload(non_hydro)
import non_hydro as model

def perturb_amp(k,ms=0.0,mb=0.0):
    # Determine critical stability parameter
    S0 = stream.Scrit(k,mb=mb)
    print("S0",S0)

    # Create class for leading order term
    h11=stream.StreamFun(k,order=1,harmonic=1,ms=ms,mb=mb)

    # Solve leading order term for coefficients
    h11.solve(S0,rhs=None)

    # Create a running dictionary of terms
    funcs={'h11':h11}

    # Solve for second order term psi22
    h22=stream.StreamFun(k,order=2,harmonic=2,ms=ms,mb=mb)
    rhs22 = model.rhs22(k,S0,funcs)
    h22.solve(S0,rhs22)

    funcs['h22']=h22

    # Second order mean field term, this term is hydrostatic
    rhs20=model.rhs20(k,S0,funcs)
    h20=stream.StreamFun(k,order=2,harmonic=0,ms=ms,mb=mb)
    h20.solve(S0,rhs20)
    h20.stress=-4*S0*h20.h
    funcs['h20']=h20


    # Third order term psi33
    h33=stream.StreamFun(k,order=3,harmonic=3,ms=ms,mb=mb)
    rhs33=model.rhs33(k,S0,funcs)
    h33.solve(S0,rhs33)
    funcs['h33']=h33


    # Third order term psi31, needs to be treated differently
    rhs31=model.rhs31(k,S0,funcs)
    h31=stream.StreamFun(k,order=3,harmonic=1,ms=ms,mb=mb)
    S2=h31.solveS(S0,rhs31,funcs,S2=0.0)
    print(funcs)
    h31.set_amp_eqn(S0,rhs=rhs31,funcs=funcs,S2=None)
    h31.s,h31.b=0,0
    funcs['h31']=h31

    rhs42=model.rhs42(k,[S0,S2],funcs)
    h42=stream.StreamFun(k,order=4,harmonic=2,ms=ms,mb=mb)
    h42.solve(S0,rhs42)
    funcs['h42']=h42

    rhs40=model.rhs40(k,[S0,S2],funcs)
    h40=stream.StreamFun(k,order=4,harmonic=0,ms=ms,mb=mb)
    h40.solve(S0,rhs40)
    h40.stress = -4*S0*h40.h - 4*S2*h20.h
    funcs['h40']=h40


    #rhs51=model.rhs51(k,[S0,S2],funcs)
    #h51=stream.StreamFun(k,order=1)
    #S4=h51.solveS(S0,rhs51,funcs,S2=S2)
    return h11,h20,h22,h33,h40,h42,funcs
