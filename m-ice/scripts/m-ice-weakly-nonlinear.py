"""
Initial version of code that tests weakly nonlinear resiults
"""

from importlib import reload
import sys
sys.path.append("../code/")

import meshModel
reload(meshModel)
from meshModel import *

import material
reload(material)
from material import *

from scipy.special import erfc


import tempModel
reload(tempModel)
from tempModel import *

import stokes2Dve
reload(stokes2Dve)
from stokes2Dve import *


from leopart import (
    particles,
    RandomRectangle,
    l2projection,
    advect_rk3,
    assign_particle_values,
    AddDelete,
    RandomCell
)

import time

import os


path  = '../code/'
path  = '../../Code/'
#path  = '/Users/jbassis/Documents/workspace/testing/Code'
sys.path.append(path)
#sys.path.append('/Users/jbassis/Documents/workspace/weakly_nonlinear/Code/')
#sys.path.append('/Users/jbassis/Documents/workspace/testing/Code')
import stress_stream
reload(stress_stream)
import stress_stream as stream
import amp_equation_tools;reload(amp_equation_tools);import amp_equation_tools as amp_tools


import matplotlib
import matplotlib.pyplot as plt
plt.ion()
plt.isinteractive()

import logging
logging.getLogger('FFC').setLevel(logging.ERROR)
logging.getLogger('UFL').setLevel(logging.ERROR)
parameters['ghost_mode']='shared_facet'

set_log_level(50)
set_log_active(False)

# Turn on plastic failure
plastic =  False


# Surface and bottom temperatures
Ts = -20.0
Tb = -20.0
xyield_min = 0.0e3
cut_off = 0e3
melange = False
buttressing = 1e-32
buttressing_height_max = 25.0 # In meters
buttressing_height_min = 55.0 # In meters

yield_strength= 0.2e6*1e12
shear_strength= 1e12
strain_rate = 0.014*0.125/1.5

fname_base = '../data/ice-shelf/ice-shelf_sims/no_melange/'

order = 3
ice_thick=400.0
k = np.pi/ice_thick/2
S0 = 1.0
amp = 0.25
amp=0.15
melt_non_dim = 1.0
evolve_k = True
crit_flag =  False
length = 4*pi/k
n=3
E=1


#length = 24*ice_thick
d = np.linspace(0,2*length,201)
surf =  ice_thick*(1-910/1020)*np.ones(np.shape(d))
bed =  -ice_thick*910/1020-1000.0*np.ones(np.shape(d))
base = surf - np.minimum(surf-bed,surf/(1-910/1020))
GL = (surf-bed)>0
Dgl = d[GL][-1]
water_depth = ice_thick*910/1020

Bedgl=bed[GL]
Basegl=base[GL]
Surfgl=surf[GL]
up_stream = Dgl-length-cut_off
Dgl = d-up_stream 
filter = (Dgl>0.0) & (Dgl <length)
Dgl = d[filter]
Bedgl = bed[filter]
Basegl = base[filter]
Surfgl = surf[filter]
offset = 0.0
d = d - offset#-up_stream
Dgl = Dgl-offset

#print("cliff height",Surfgl[-1])


# Set mesh resolution and estimate approximate number of points in x/z dir
ice_thick = Surfgl[0]-Basegl[0]

# Define geometry of domain
left_vel = -0*0.5*strain_rate*length/material.secpera*material.time_factor
right_vel = 2*0.5*strain_rate*length/material.secpera*material.time_factor


#amp = 0.1*ice_thick

#h11,h20,h22,h33,h40,h42,funcs=perturb.perturb_amp(k*ice_thick,mb=melt_non_dim)
#Scrit,funcs,amp_eqn1=amp_tools.solveAmpEqn(k*ice_thick,mb=melt_non_dim,n=n,E=1);
#amp_eqn=amp_tools.AmpEqn(k*ice_thick,Scrit[0],melt=0,n=3,order=3)
amp_eqn=amp_tools.AmpEqn(k*ice_thick,S0,melt=melt_non_dim*0,n=3,crit=True)
funcs=amp_eqn.funcs
h11=funcs['h11']
h22=funcs['h22']
h20=funcs['h20']
h31=funcs['h31']
h33=funcs['h33']
h40=funcs['h40']
h42=funcs['h42']
h44=funcs['h44']
h51=funcs['h51']
#h42.s=h42.s
#h42.b=h42.b
#h44.s=h44.s
#h44.b=h44.b
#h31.s=h31.s
#h31.b=h31.b
print(h11.s,h22.s,h20.s,h22.s,h31.s,h33.s,h40.s,h42.s,h44.s)
dh = (h11.s-h11.b)*amp+(h22.s-h22.b)*amp**2+(h33.s-h33.b)*amp**3+(h42.s-h42.b)*amp**4
#print('Total perturb',dh)
lam = 2*np.pi/k
def bot_fun(xx):
   x = xx-lam/2#-ice_thick
   b11 = h11.b*amp
   b22 = h22.b[0]*amp**2
   b33 = h33.b[0]*amp**3
   b31 = h31.b*amp**3
   b42 = h42.b[0]*amp**4*(order>3)
   b44 = h44.b[0]*amp**4*(order>3)
   b51 = h51.b*amp**5*(order>3)
   b = -stream.rho_i/stream.rho_w*ice_thick  + ((b11+b31+b51)*np.cos(k*x) + (b22+b42)*np.cos(2*k*x) + b33*np.cos(3*k*x)+b44*np.cos(4*k*x))*ice_thick
   return b

def surf_fun(xx):
   x = xx-lam/2#-ice_thick
   s11 = h11.s*amp
   s22 = h22.s[0]*amp**2
   s33 = h33.s[0]*amp**3
   s31 = h31.s*amp**3
   s42 = h42.s[0]*amp**4*(order>3)
   s44 = h44.s[0]*amp**4*(order>3)
   s51 = h51.s*amp**5*(order>3)
   s = (1-stream.rho_i/stream.rho_w)*ice_thick + ((s11+s31+s51)*np.cos(k*x)+(s22+s42)*np.cos(2*k*x)+s33*np.cos(3*k*x)+s44*np.cos(4*k*x))*ice_thick
   return s

#print('****')
#print(h11.b,h22.b,h33.b,h42.b,h44.b)
#print('***')
#bot_fun = interp1d(d,base,fill_value=(base[0],base-0.4291911130781180844410915035478[-1]),bounds_error=False)
bed_fun = interp1d(d,bed,fill_value=(bed[0],bed[-1]),bounds_error=False)
bed_fun_np = interp1d(d,bed,fill_value=(bed[0],bed[-1]),bounds_error=False)
#surf_fun = interp1d(d,surf)
ice_thick_fun = interp1d(d,surf-base)

#dz = round(ice_thick/13.333333333/2)
dz = round(ice_thick/13.333333333/2*2)
Nx = int(length/dz)
Nz = int(ice_thick/dz)
fname_base = fname_base + str(strain_rate)+'/'+str(dz) + '/'

# Initialize mesh
start = time.time()
mesh = MeshModelPoly(surf_fun,bot_fun,bed_fun,Nx=Nx,Nz=Nz,length=length,dz=dz)
#mesh = MeshModel(surf_fun,bot_fun,bed_fun,Nx=Nx,Nz=Nz,length=length,dz=dz)
mesh.length = length
print('Time to make mesh and basis functions',time.time()-start)

#_____________________________________________
# Create particles defined on mesh
#xm,zm = mesh.get_coords();xmin = np.min(xm); xmax=np.max(xm);ymin = np.min(zm); ymax=np.max(zm)
#xp = RandomRectangle(Point(xmin, ymin), Point(xmax, ymax)).generate([Nx*10, Nz*10])
p_min = 8 
p_max = 16

p_min = 16
p_max = 32
print('Generating particles')
gen = RandomCell(mesh.mesh)
xp = gen.generate(p_min)
print('Done generating particles')
#_____________________________________________
# Define function space for strain and temperature function
Vdg = FunctionSpace(mesh.mesh, 'DG',1)

print('Initializing function spaces')
strain_mesh, temp_mesh, epsII_mesh = Function(Vdg), Function(Vdg), Function(Vdg)

# Initial conditions for strain and temp
strain_fun = Expression("0.0", degree=1)
temp_fun = temp_init(Ts,Tb,surf_fun, bed_fun,degree=1)

# Create functions defined on mesh with initial values
strain_mesh.assign(strain_init)
temp_mesh.assign(interpolate(temp_fun,Vdg))
epsII_mesh.assign(strain_init)

# Particle values at nodes
pstrain = assign_particle_values(xp, strain_fun)
pepsII = assign_particle_values(xp, strain_fun)
ptemp = assign_particle_values(xp, temp_fun)
print('Done initializing function spaces')


# Now we initialize the particle class
print('Creating particles')
p = particles(xp, [pstrain,ptemp,pepsII], mesh.mesh)
print('Done particles')


# Make sure we have enough particles per cell
#AD = AddDelete(p, p_min, p_max, [strain_mesh, temp_mesh,epsII_mesh]) # Sweep over mesh to delete/insert particles
#AD.do_sweep()

(xp , pstrain , ptemp, pepsII) = (p. return_property(mesh , 0) ,
    p. return_property(mesh , 1) ,
    p. return_property(mesh , 2),
    p. return_property(mesh , 3))
#_____________________________________________
# Initialize temperature model
print('Initializing temperature')
Tmodel = tempModel(mesh.mesh,Tb=Tb,Ts=Ts)
x = SpatialCoordinate(mesh.mesh)
#zb = bot_fun(x[0])%
Tmodel.set_temp(x,surf_fun,bot_fun,bed_fun)
print('Done initializing temperature')

#_____________________________________________
# Viscosity and material properties
# Define the rheology and make sure we set some of the properties

glenVisc = glenFlow(grain_size=0.5)
glenVisc.yield_strength=yield_strength#yield_strength
glenVisc.shear_strength=shear_strength
glenVisc.visc_min = 1e7
glenVisc.visc_max = 1e18
glenVisc.plastic = plastic
glenVisc.yield_min = 5e3
glenVisc.crit_strain = 0.1
glenVisc.mu = 1.0
glenVisc.strain_crit = 0.1/4

fname_base = fname_base + str(yield_strength)+'/'

#_____________________________________________
# Viscosity and material properties
# Set inflow velocity of the domain

#_____________________________________________
# Define our model and some parameters
model = Stokes2D(mesh,glenVisc,left_vel=left_vel,right_vel=right_vel)
model.tempModel=Tmodel
model.tolerance = 1e-5
model.u_k = None
model.maxit = 25
model.maxit_local = 25
model.local_err_min = 1.0
model.tracers = particles

#model.calving_front = False # This applies a stress boundary condition to the right edge
model.alpha = 1.0  # Not used anymore, legacy from when we used overrelaxation
model.water_drag = 0.0 # Quadratic drag term associated with water drag
# Add lateral drag to model
B = 0.75e8
width = 10e3
model.lateral_drag = 0.0#2*(4)**(1.0/3)*B/(width**(4.0/3))*0
model.bed_yield_strength = 100e3
model.friction = 0.0#4.8e5#/(surf_fun(0)-bed_fun(0))/model.rho_i/model.g # Friction coefficient
model.m = 1.0/3.0 # Friction exponent

model.left_wall = 0.0
model.right_wall = length 
model.right_vel=right_vel
#_____________________________________________
# Maximum time step
time_step_secs =  86400.0*30# Time step in seconds
#time_step_secs = 86400*365*10
#time_step_secs = 86400*56
time_step_secs = 86400.0*10000#/16*8# Time step in seconds
#time_step_secs = 86400.0*30#/16*8# Time step in seconds
#time_step_secs = 86400/100
time_step = time_step_secs/material.time_factor # Convert time step to unit we are using


#_____________________________________________
# Start loop for simulation
# Initialize time vector to zero
t = 0.0
it_type = 'Picard'

max_length = 10*length# Regrid if length exceeds this value
min_length = length # Set new length after regridding to this value
model.mesh.length = length # Set this as the max length of the mesh--doesn't actually do anything


#max_length =length
#min_length =length
model.mesh.length =length
save_files = False # Set to True if we want to save output files

if save_files==True:
    if not os.path.exists(fname_base):
    	os.makedirs(fname_base)
    import shutil
    script_name = 'm-ice-shelf.py'
    shutil.copy2(script_name, fname_base+script_name)
    shutil.copy2('../m-ice/stokes2Dve.py', fname_base+'stokes2Dve.py')


CFL = 0.5
model.u_k = None
i =0
model.strain = Function(model.mesh.Q)
L2_strain = []
tlist = []
model.method = 1
model.buttressing = buttressing # Apply buttressing force (in Pa) in x-direction to portion of ice under water???
model.buttressing_height_max = buttressing_height_max # Apply buttressing force (in Pa) in x-direction to portion of ice under water???
model.buttressing_height_min = buttressing_height_min # Apply buttressing force (in Pa) in x-direction to portion of ice under water???

model.buttressing = 1e-32
print("Name of directory",fname_base)
mu=glenVisc.ductile_visc(strain_rate,Ts+273.15)
B=2*mu*strain_rate**(1-1/3)
#S0=model.rho_i*model.g*(1-model.rho_i/model.rho_w)*ice_thick/(4*0.5*B*strain_rate**(1/3))
#print(S0)
#tau_xx=mu*strain_rate
#S0 = 910*9.81*(1-910/1020)*ice_thick/4/tau_xx
xx_init=np.linspace(0,length,1001)
surf_init = model.mesh.surf_fun(xx_init)
bot_init = model.mesh.bot_fun(xx_init)
sea_level_offset = np.min(surf_init)
model.sea_level_offset = np.minimum(sea_level_offset,0.0)
mean_thick = np.mean(surf_init-bot_init) 
mean_thick = ice_thick
H0 = mean_thick
strain_rate=(model.rho_i*model.g*(1-model.rho_i/model.rho_w)*mean_thick/(4*0.5*B*S0))**3
#print(S0)
model.left_vel = -0*0.5*strain_rate*length/material.secpera*material.time_factor
model.right_vel = 2*0.5*strain_rate*length/material.secpera*material.time_factor
dam = []
tt = []
model.calving_front=False

model.bot_melt = melt_non_dim*ice_thick*strain_rate
model.surf_melt =  0.0
xx=np.linspace(0,length,1001)
xx_orig = np.linspace(0,length,501)
#print('h11',np.max((amp*h11.s-amp*h11.b)*np.cos(k*xx)))
damage=stream.amp2dam(k*ice_thick,[amp],funcs)[0]


temp = -20+273
mu=glenVisc.ductile_visc(strain_rate,-20+273)
epsII = strain_rate
Bdiff =  glenVisc.pre_diff*np.exp(glenVisc.E_diff/(8.314*temp))
Bdisl = (glenVisc.pre1*np.exp(glenVisc.E1/(3.0*8.314*temp))*(temp<=263.15) + glenVisc.pre2*np.exp(glenVisc.E2/(3.0*8.314*temp))*(temp>263.15))
visc_glen = (epsII**(2.0/3.0)/Bdisl)**(-1)
visc_newt = Bdiff
#n = 3
#n_eff = (1+(1-n)/n*visc_newt/(visc_newt+visc_glen))**(-1)
#print('****** effective flow law exponent ****',n_eff)

kkk3,t_3,dam_3=amp_tools.evolveK(k*ice_thick,S=S0,A=amp,melt_rate=melt_non_dim,dt=0.005,Nt=800,n=3,crit=False,order=3,evolve_k=evolve_k)
kkk2,t_2,dam_2=amp_tools.evolveK(k*ice_thick,S=S0,A=amp,melt_rate=melt_non_dim,dt=0.005,Nt=800,n=2,crit=False,order=3,evolve_k=evolve_k)
kkk1,t_1,dam_1=amp_tools.evolveK(k*ice_thick,S=S0,A=amp,melt_rate=melt_non_dim,dt=0.005,Nt=800,n=1,crit=False,order=3,evolve_k=evolve_k)
kk2,t_2,dam_2=amp_tools.evolveK(k*ice_thick,S=S0,A=amp,melt_rate=melt_non_dim,dt=0.005,Nt=800,n=3,crit=False,order=1,evolve_k=evolve_k)
#kkk1,t_3_proj,dam_3_proj=amp_tools.evolveK(k*ice_thick,S=S0,A=amp,melt_rate=melt_non_dim,dt=0.005,Nt=800,n=n,crit=False,order=3,evolve_k=evolve_k)
#kkk2,t_3_crit,dam_3_crit=amp_tools.evolveK(k*ice_thick,S=S0,A=amp,melt_rate=melt_non_dim,dt=0.005,Nt=800,n=n,crit=True,order=3,evolve_k=evolve_k)
#kkk3,t_5_crit,dam_5_crit=amp_tools.evolveK(k*ice_thick,S=S0,A=amp,melt_rate=melt_non_dim,dt=0.005,Nt=800,n=n,crit=True,order=5,evolve_k=evolve_k)
#kkk4,t_5_proj,dam_5_proj=amp_tools.evolveK(k*ice_thick,S=S0,A=amp,melt_rate=melt_non_dim,dt=0.005,Nt=800,n=n,crit=False,order=5,evolve_k=evolve_k)

damage = 1-np.min(surf_fun(xx)-bot_fun(xx))/np.mean(surf_fun(xx)-bot_fun(xx))
#damage = 1-np.min(surf_fun(xx)-bot_fun(xx))/np.max(surf_fun(xx)-bot_fun(xx))

print("Initial damage",damage,'dimensionless melt rate',melt_non_dim,'dimensional melt',model.bot_melt)
dam.append(damage)
tt.append(0.0)
tt_real = []
time_real = 0.0
tt_real.append(time_real)
model.strain_rate=strain_rate
print('Non-dimensional time step',time_step*strain_rate)
bot_list = []
bot_list2 = []
#for i in range(i,100000):
L = length
left = 0.0
for i in range(i,2500):
   #L2_strain.append(assemble(model.strain*dx(model.mesh.mesh))/assemble(Constant(1.0)*dx(model.mesh.mesh)))
   #L2_strain.append(assemble(model.strain*dx(model.mesh.mesh)))
   #tlist.append(t)
   # First need to interpolate tracer quantities to nodes
   #node_vars = particles.tracers_to_nodes()

   # For first time step, we use a small time step.  After that, we use the CFL criterion
   if i==0:
       time_step = time_step_secs/material.time_factor
   else:
       Q0 = FunctionSpace(model.mesh.mesh, "DG", 0)
       quality=np.min(MeshQuality.radius_ratios(model.mesh.mesh).array())
       time_step = CFL*np.min(project(CellDiameter(model.mesh.mesh)/sqrt(inner(u,u)),Q0).compute_vertex_values())
       time_step = np.minimum(time_step_secs/material.time_factor,time_step)


   #print('Time step',time_step)
   u,pres = model.solve(p,dt=time_step,tolerance=model.tolerance)
   Q1 = FunctionSpace(model.mesh.mesh, "DG", 1)
   epsilon = 0.5*(nabla_grad(u) + nabla_grad(u).T)
   tau = project(model.eta*epsilon[0,0]-pres,Q1)

   # Do a little bit of accounting to make sure that our time step doesn't violate the CFL criterion
   ux, uz = model.get_velocity();speed = np.sqrt(ux**2+uz**2)
   Q0 = FunctionSpace(model.mesh.mesh, "DG", 0)
   time_step_CFL = CFL*np.min(project(CellDiameter(model.mesh.mesh)/sqrt(inner(u,u)),Q0).compute_vertex_values())
   #print('Time step CFL',time_step_CFL)
   #if time_step_CFL<0.5*time_step:
    #   time_step=time_step_CFL
     #  u,pres = model.solve(node_vars,dt=time_step,tolerance=model.tolerance);#model.u_k = None
   #xm,zm = particles.get_coords()

   #particles.tracers['Strain'][xm<1e3]=particles.tracers['Strain'][xm<1e3]/(1.0+time_step/tau)

   # Decide if we need to remesh
   x,z = model.mesh.get_coords()
   if np.max(x)<max_length:
       remesh_elastic=False
       #model.mesh.length = max_length
   else:
       remesh_elastic=True
       #model.mesh.length=length

   if model.mesh.mesh.hmax()/model.mesh.mesh.hmin() > 10.0:
       remesh_elastic=False
       #model.mesh.length=min_length

   quality=np.min(MeshQuality.radius_ratios(model.mesh.mesh).array())
   #if quality<0.2:
   #    model.mesh.length=min_length
   #    remesh=True
   #else:
   #    remesh=False

   remesh=True
   #remesh_elastic = False
   remesh_elastic = True
   remesh = True
   model.mesh.length=10*max_length
   #print(remesh_elastic)
   #print(remesh)
   if np.mod(i,1)==0:
      #particles.tracers['Strain'][xm<3.5e3]=0.0
      if save_files == True:
          #uxm = particles.tracers['ux']
          #uzm = particles.tracers['uz']
          (xp , pstrain , ptemp, pepsII) = (p. return_property(mesh , 0) ,
            p. return_property(mesh , 1) ,
            p. return_property(mesh , 2),
            p. return_property(mesh , 3))

          #eta_visc_m = model.tracers.nodes_to_tracers(model.eta_visc)
          #eta_plas_m = model.tracers.nodes_to_tracers(model.eta_plas)
          #eta_m = model.tracers.nodes_to_tracers(model.eta)
          fname =fname_base + 'glacier_cliff_'+str(i).zfill(3)+'.npz'
          print(fname)
          np.savez(fname, t=t, xm=xp[:,0],zm=xp[:,1],speed=speed,ux=ux,uz=uz,strain=pstrain,
               epsII=pepsII/material.time_factor,temp=ptemp)

          mesh_file_name =fname_base + 'glacier_cliff_'+str(i).zfill(3)+'.xml'
          mesh_file = File(mesh_file_name)
          mesh_file << model.mesh.mesh
          temp_file_name = fname_base + 'temp_'+str(i).zfill(3)+'.hdf'
          temp_file=HDF5File(model.mesh.mesh.mpi_comm(), temp_file_name, 'w')
          temp_file.write(u,'u')
          temp_file.write(model.strain,'strain')
          temp_file.close()
   remesh_elastic=False
   #model.calving_front=True
   
   # Switched off for testing
   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   #length = length + time_step*(model.right_vel-model.left_vel)
   model.mesh.length=length
 

   #model.mesh.length =12*ice_thick
   # Update all quantities (need to update this to simplify it and use RK4 if specified)
   time_step = model.update(u,time_step,p,remesh=remesh,remesh_elastic=remesh_elastic)
   yr = int(np.mod(t,365))
   #day = round((t - yr)*365,1)
   day = (t - yr)*365
   hr,day = np.modf(day)
   hr = round(hr*24,0)
   title_str = 'Time: '+str(yr).zfill(1)+ 'a '+str(int(day)).zfill(1)+'d '+str(int(hr)).zfill(1)+'hr'
   title_str = 'Time '+str(t).zfill(3)


   if remesh_elastic == False:

       Vdg = FunctionSpace(model.mesh.mesh, 'DG',1)
       Vcg = FunctionSpace(model.mesh.mesh, 'DG',1)
       (xp , pstrain , ptemp, pepsII) = (p. return_property(mesh , 0) ,
          p. return_property(mesh , 1) ,
          p. return_property(mesh , 2),
          p. return_property(mesh , 3))
       del p
       p = particles(xp, [pstrain,ptemp,pepsII], model.mesh.mesh)
   else:
       p.relocate()

   # Advect particles -- Turn this on to advect particles now that it is removed from Stokes script
   Vdg = FunctionSpace(model.mesh.mesh, 'DG',1)
   ap = advect_rk3(p, model.vector2, model.u_k, "open")
   ap.do_step(time_step)
   AD = AddDelete(p, p_min, p_max, [interpolate(model.strain,Vdg), interpolate(model.temp,Vdg) , interpolate(model.epsII,Vdg)]) # Sweep over mesh to delete/insert particles
   AD.do_sweep()
   # Plotting
   (xp , pstrain , ptemp, pepsII) = (p. return_property(mesh , 0) ,
       p. return_property(mesh , 1) ,
       p. return_property(mesh , 2),
       p. return_property(mesh , 3))
   pstrain[xp[:,0]<xyield_min]=0.0
   pstrain[xp[:,0]>(length-xyield_min)]=0.0
   pstrain = np.minimum(pstrain,0.1)
   p.change_property(pstrain,1)
   # Print some diagnostics to screen for debugging purpose
   xx=np.linspace(0,length,1001)
   surf = model.mesh.surf_fun(xx)
   bot = model.mesh.bot_fun(xx)
   #filter = (xx>mean_thick) & (xx<length-mean_thick)
   perturb = np.min(surf-bot)
   #damage = 1-perturb/mean_thick
   #mean_thick = np.mean(surf-bot) 
   mean_thick = mean_thick-(strain_rate*mean_thick+melt_non_dim*strain_rate*mean_thick)*time_step
   #mean_thick = mean_thick*np.exp(-(strain_rate+melt_non_dim*strain_rate*mean_thick)*time_step)
   
   H0 += -strain_rate*H0*time_step - model.bot_melt*time_step
   #print('****mean thick',mean_thick)
   #print('***min thick',np.min(surf-bot))
   #print('***damage',damage)
   damage = 1-np.min(surf-bot)/np.mean(surf-bot)
   #damage = np.max(surf-bot)/mean_thick-1
   #idx=np.argsort(surf-bot)
   #h = surf-bot
   #thick_min = h[idx[0]]
   #thick_max1 = h[idx[-1]]
   #thick_max2 = h[idx[-2]]
   #thick_mean = 0.5*(thick_max1+thick_max2)
   #damage=1-thick_min/thick_mean
   t = t+time_step*strain_rate
   time_real = time_real + time_step
   tt.append(t)
   tt_real.append(time_real)
   dam.append(damage)
   #k = k*np.exp(-time_step*strain_rate*(E+0*melt_non_dim))
   left += -2*(model.left_vel-model.right_vel)*time_step
   #lam = 2*np.pi/k

   amp_eqn=amp_tools.AmpEqn(k*mean_thick,S0,melt_non_dim,n=n,crit=crit_flag,E=1)
   dAdt = amp_eqn(S0,amp,order=order).item()
   amp += dAdt*time_step*strain_rate
   #xx=np.linspace(0,length,501)
   #xs=np.linspace(0,length,101)
   xxx,surf_a,bot_a=stream.topo(k*mean_thick,amp,funcs,order=3)
   plt.figure(1);plt.clf();#plt.ion()
   plt.subplot(3,1,1)
   plot(model.mesh.mesh)
   plt.plot((xxx*mean_thick-lam/2)*np.exp(strain_rate*time_real),surf_a*mean_thick,'k')
   plt.plot((xxx*mean_thick-lam/2)*np.exp(strain_rate*time_real),bot_a*mean_thick,'k')
   plt.plot([0,length],[0,0],'k')
   plt.plot([0,length],[-25,-25],'k')
   #plt.plot(xx_init,surf_init)
   #plt.plot(xx_init,bot_init)
   #plt.plot(xx,bot,'--')
   #plt.plot(xx,surf,'--')
   #mean_thick = np.mean(surf-bot) 
   #plt.plot(xx,H0*np.ones_like(xx)*(1-model.rho_i/model.rho_w),'k')
   #plt.plot(xx,-H0*np.ones_like(xx)*model.rho_i/model.rho_w,'k')
   plt.ylim((-400,100))
   plt.xlim((0,length))
   #plt.plot(model.xsurf,model.zsurf,'.r')
   #plt.plot(xx_orig,bot_fun(xx_orig),'--r')
   plt.subplot(3,1,2)
   plot(u)
   plt.plot(xx,bot)
   plt.plot([0,length],[0,0],'k')
   plt.subplot(3,1,3)
   bot_list.append(-H0*model.rho_i/model.rho_w)
   bot_list2.append(np.mean(bot))
   #plt.plot(tt[1::],np.array(bot_list),'ok')
   #plt.plot(tt[1::],np.array(bot_list2),'.r')
   #plt.plot(t_3_crit,dam_3_crit,'--r',zorder=3,label='crit 3')
   plt.plot(t_3,dam_3,'-r',zorder=4,label='n=3')
   plt.plot(t_1,dam_1,':k',zorder=5,label='n=1')
   plt.plot(t_2,dam_2,'-.',zorder=6,label='linear')
   #plt.plot(t_2,dam_2[0]*np.exp((3*(1-S0)+melt_non_dim)*np.array(t_2)))
   #plt.plot(t_5_crit,dam_5_crit,'--b',zorder=5,label='crit 5')
   #plt.plot(t_5_proj,dam_5_proj,color='c',linestyle=':',zorder=1,label='proj 5')
   plt.legend()
   #plt.plot(np.array(tt[::]),np.array(dam[::]),'k');
   plt.plot(np.array(tt[1::]),np.array(dam[1::]),'.k');
   plt.ylim([0,1])
   plt.xlim([0,1])
   #plt.xlim([0,np.max(tt)])
   plt.title(title_str)
   plt.show()
   plt.draw()
   plt.pause(1e-16)

   #plt.xlim([0,length])
   #ax2=plt.subplot(2,1,2)
   #c=plt.scatter(xp[:,0],xp[:,1],s=0.1,c=np.log10(pepsII/material.time_factor+1e-16),vmin=-8,vmax=-6);cbar2=plt.colorbar(c);
   #plt.axis('equal')
   #plt.plot(xx,bed_fun_np(xx),'--k',linewidth=2)
   #plt.plot(xs,surf_fun(xs),'--',color='gray')
   #cbar2.set_ticks([-8,-6])
   #plt.title(title_str)
   #plt.xlim([0,length])
   #plt.xlabel('Distance (km)')
   #for ax in [ax1,ax2]:
   #    ax.set_xticks([])
   #    ax.set_yticks([])
   #    ax.spines['right'].set_visible(False)
   #    ax.spines['top'].set_visible(False)
   #    ax.spines['left'].set_visible(False)
   #    ax.spines['bottom'].set_visible(False)
   #    ax.plot(xx,bot_fun(xx),color='brown',linewidth=2)
   #    #ax.plot(xx,bed_fun_np(xx),'--k',linewidth=2)
   #    ax.plot(xs,surf_fun(xs),'--',color='gray')
   #    if melange == True:
   #        plt.gca()
   #        plt.axvline(model.right_wall,linestyle='--',color='k')
   #ax2.spines['bottom'].set_visible(True)
   #ax2.set_xticks([0,3e3,10*ice_thick,max_length])
   ##ax2.set_xticklabels([0,3,10*ice_thick/1e3,max_length/1e3])
   #plt.xlim([0,length])

   #ax1.spines['bottom'].set_visible(True)
   #ax1.set_xticks([0,3e3,10*ice_thick,max_length])
   #ax1.set_xticklabels([0,3,10*ice_thick/1e3,max_length/1e3])
   #plt.xlim([0,length])
   #plt.pause(1e-16);
   #fig=plt.figure(1)
   #fig.clf()
   #ax=plot(sqrt(dot(u,u)));plt.colorbar(ax)
   #plt.title(title_str)
   #plt.pause(1e-1)
   #plt.draw()
   #plt.ion()
   #plt.show()
   #print('Time step',time_step,'strain rate',strain_rate,'Mesh quality',model.mesh.mesh.hmax()/model.mesh.mesh.hmin(),'quality ratios',quality,'number of negative epsII',sum(pepsII<0),'Percent yielded',np.sum(pstrain>0)/len(pstrain)*100,'Maximum strain',np.max(pstrain))
   #surf = model.mesh.surf_fun(xx)
   #bot = model.mesh.bot_fun(xx)
   #perturb = np.min(surf-bot)
   #if perturb<0:
   #    break
   #mean_thick = np.mean(surf-bot) 
   #damage = 1-perturb/mean_thick
   #print("****Perturbation amplitude",np.min(surf)-np.max(bot),'Mean thickness',np.mean(surf-bot),"Damage",1-perturb/mean_thick)
   mu=glenVisc.ductile_visc(strain_rate,-20+273)
   #mu = 0.5*B*strain_rate**(1/3-1)
   #S0=model.rho_i*model.g*(1-model.rho_i/model.rho_w)*mean_thick/(0.5*B*strain_rate**(1/3))
   #strain_rate=(model.rho_i*model.g*(1-model.rho_i/model.rho_w)*mean_thick/(0.5*B*S0))**3
   #strain_rate= (model.rho_i*model.g*(1-model.rho_i/model.rho_w)*mean_thick/(4*0.5*B*S0))
   #S0=model.rho_i*model.g*(1-model.rho_i/model.rho_w)*mean_thick/(4*0.5*B*strain_rate**(1/3))
   #print(S0)
   #tau_xx=mu*strain_rate
   #S0 = 910*9.81*(1-910/1020)*mean_thick/4/tau_xx
   #print(S0)
   #print(strain_rate)


   # Adjust S0 to keep it consistent with new length and ice thickness
   strain_rate= (model.rho_i*model.g*(1-model.rho_i/model.rho_w)*mean_thick/(4*0.5*B*S0))**3
   S0=model.rho_i*model.g*(1-model.rho_i/model.rho_w)*mean_thick/(4*0.5*B*strain_rate**(1/3))
   model.strain_rate=strain_rate


   #print(strain_rate)
   #model.mesh.length=12*length
   c=model.mesh.mesh.coordinates()
   length=np.max(c[:,0])-np.min(c[:,0])
   model.right_wall=np.max(c[:,0])
   model.left_wall= np.min(c[:,0])
   model.mesh.length = model.right_wall-model.left_wall
   length = model.mesh.length
   model.left_vel =  -0*0.5*strain_rate*length/material.secpera*material.time_factor
   model.right_vel =  2*0.5*strain_rate*length/material.secpera*material.time_factor
   #model.left_vel=left_vel
   #model.right_vel=right_vel
   print('*******************************************')
   print("S0",S0,"Initial amplitude",amp,"Perturbation amplitude",np.min(surf-bot)-mean_thick,'Mean thickness',mean_thick,"Damage",damage,'thinning factor',mean_thick/ice_thick)
   #model.left_vel=left_vel
   #model.right_vel=right_vel

   # Print some diagnostics to screen for debugging purpose
   #t = t+time_step
   #tt.append(t)
   #dam.append(damage)
   if i>1:
   	poly_nom= np.polyfit(tt[1::], dam[1::],1) 
   else:
        poly_nom= np.polyfit(tt[::], dam[::],1) 
   print('Time:  ',t*material.time_factor/material.secpera,'Time step',time_step)
   print('Damage slope',poly_nom[0])
   print('*******************************************')
   c=model.mesh.mesh.coordinates()
   #print('Max x',np.max(c[:,0]),'Min x',np.min(c[:,0]))
   
   # Switched off for testing
   length=np.max(c[:,0])-np.min(c[:,0])
   model.right_wall=np.max(c[:,0])
   model.left_wall= np.min(c[:,0])
   model.mesh.length = model.right_wall-model.left_wall
   length = model.mesh.length
   model.bot_melt =  melt_non_dim*mean_thick*strain_rate
   model.surf_melt = 0.0
   #model.accum=strain_rate*mean_thick*0
   model.strain_rate = strain_rate
   #model.mesh.length=10*length
   if t>1.0:
       break
   if damage>0.95:
       break
   if mean_thick/ice_thick<0.1:
       break
#if model.method==1:
#    plt.figure(4);plt.plot(tlist,L2_strain,label=time_step_secs)
#else:
#    plt.figure(4);plt.plot(tlist,L2_strain,'--',label=time_step_secs)
#plt.legend()

#mu=glenVisc.ductile_visc(strain_rate,-20+273)
#tau_xx=mu*strain_rate**(1/3)

