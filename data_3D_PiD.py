import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spint
import matplotlib

inputfile = "data/Gridded_v_p_093000_improvPStats_withPiD.npz"

data = np.load(inputfile)

data = {k:v for k,v in data.items()}
print(data.keys())
vx = data['vx']
pdyn = data['pdyn']
pid = data['PiD']
ps = data['PressureStrain']
pth = data['PressureDilatation']


# Yan et al datasets
PS_innertetra = np.loadtxt("/home/mjalho/PO/amitis-data-interface/data/psterms_tetra_1567.csv", skiprows=1, delimiter=',')
sampled_path = np.loadtxt("/home/mjalho/PO/amitis-data-interface/data/sampled_v_p_along_tracks_aligned.csv", skiprows=1, delimiter=',')

# print(PS_innertetra)
# print(sampled_path)

# let's plot a central slice of the data in the Z direction

plt.imshow(vx[:,:,20],origin="lower")

plt.savefig("vtot_xy_plane_test.png", dpi=300)
plt.close()

'''
readme.txt from the example data:
        Array extent:
            (Nz, Ny, Nx)=(400, 601, 400)

        Data bounds:
           z: -100000 to +100000  km
           y: -150250 to +150250  km
           x:  -50000 to +150000  km

        Cell sizes:
           (dz, dy, dx)=(500, 500, 500) km
'''
z = np.sort(np.unique(data['z']))
y = np.sort(np.unique(data['y']))
x = np.sort(np.unique(data['x']))
# print(x)
# print(x.shape, y.shape, z.shape)
# print(vx.shape)

# along vlasiator-data-interface/example_flight.py

flight = np.loadtxt("trajectories/PO_constellation_flight.txt", delimiter=',')

pts = flight[:,2:5] # convert to km from m
t = flight[::7,0]

bary = 1.0/7.0*(pts[0::7,:]+pts[1::7,:]+pts[2::7,:]+pts[3::7,:]+pts[4::7,:]+pts[5::7,:]+pts[6::7,:])

# print(bary[:,2])
intp = spint.RegularGridInterpolator( (x,y,z), pid, bounds_error=False, fill_value=np.nan)
vals = intp(pts)

fig, axs = plt.subplots(2)

# print(pts[0::7,0])

#plot outer tetra vtot
for po in [0,1,2,3]:
   axs[0].plot(t,vals[po::7], label="outer:{:d}".format(po))

#plot inner tetra vtot
for po in [0,4,5,6]:
   axs[1].plot(t,vals[po::7], label="inner:{:d}".format(po))

valsb = intp(bary)
axs[0].plot(t,valsb, color='k')
valsb = intp(bary)
axs[1].plot(t,valsb, color='k')

axs[0].set_ylabel("PiD (line)")
axs[1].set_ylabel("PiD (line)")
axs[0].set_xlim((300,700))
axs[1].set_xlim((300,700))
axs[0].legend()
axs[1].legend()

intp = spint.RegularGridInterpolator( (x,y,z), pdyn, bounds_error=False, fill_value=np.nan)
vals = intp(pts)

axs01 = axs[0].twinx()
axs11 = axs[1].twinx()

#plot outer tetra pdyn
for po in [0,1,2,3]:
   axs01.plot(t,vals[po::7], label="outer:{:d}".format(po),linestyle='dotted')

#plot inner tetra pdyn
for po in [0,4,5,6]:
   axs11.plot(t,vals[po::7], label="inner:{:d}".format(po),linestyle='dotted')


valsb = intp(bary)
axs[0].plot(t,valsb, color='k')
valsb = intp(bary)
axs[1].plot(t,valsb, color='k')

axs01.set_ylabel("Pdyn (dots)")
axs11.set_ylabel("Pdyn (dots)")



plt.savefig("flight_test.png",dpi=300)
plt.close()


t = sampled_path[::7,0]
xyz = sampled_path[:,2:5]

pts = xyz
bary = 1.0/4.0*(pts[0::7,:]+pts[4::7,:]+pts[5::7,:]+pts[6::7,:])

vals = intp(pts)
fig, axs = plt.subplots(2)

# print(pts[0::7,0])

# #plot outer tetra vtot
# for po in [0,1,2,3]:
#    axs[0].plot(t,vals[po::7], label="outer:{:d}".format(po))

#plot inner tetra PS terms
# axs[1].plot(PS_innertetra[:,0],PS_innertetra[:,1], label=r"$_\Theta$")
axs[1].plot(PS_innertetra[:,0],PS_innertetra[:,2]*1e9, label=r"$\Pi : D$")
# axs[1].plot(PS_innertetra[:,0],PS_innertetra[:,3], label=r"$PS$")

# valsb = intp(bary)
# axs[0].plot(t,valsb, color='k')
# valsb = intp(bary)
# axs[1].plot(t,valsb, color='k')

axs[0].set_ylabel(r"$P_{dyn}$ [nPa]")
axs[1].set_ylabel(r"$\Pi : D$ [nPa s$^{-1}$]")
# axs[0].set_xlim((300,700))
# axs[1].set_xlim((300,700))
axs[1].legend()

intp = spint.RegularGridInterpolator( (x,y,z), pdyn*1e9, bounds_error=False, fill_value=np.nan)
vals = intp(pts)

# axs01 = axs[0].twinx()
# axs11 = axs[1].twinx()

#plot both tetra pdyn
for po in [0,1,2,3]:
   axs[0].plot(t,vals[po::7], label="outer:{:d}".format(po))

#plot inner tetra pdyn
for po in [4,5,6]:
   axs[0].plot(t,vals[po::7], label="inner:{:d}".format(po), linewidth=1)

axs[0].legend()

# valsb = intp(bary)
# axs[0].plot(t,valsb, color='k')
intp = spint.RegularGridInterpolator( (x,y,z), pth, bounds_error=False, fill_value=np.nan)
valspth = intp(bary)
intp = spint.RegularGridInterpolator( (x,y,z), pid, bounds_error=False, fill_value=np.nan)
valspid = intp(bary)
intp = spint.RegularGridInterpolator( (x,y,z), ps, bounds_error=False, fill_value=np.nan)
valsps = intp(bary)
axs[1].set_prop_cycle(None)
# axs[1].plot(t,2*valspth, linewidth=1,label=r"$2*P_{\Theta\mathrm{sim}}$")
axs[1].plot(t,valspid*1e9, linewidth=1,label=r"$\Pi : D_\mathrm{sim}$")
# axs[1].plot(t,2*valsps,  linewidth=1,label=r"$2*PS_\mathrm{sim}$")

axs[1].legend()
matplotlib.rcParams['font.family'] = 'Arial'
# axs01.set_ylabel("Pdyn (dots)")
# axs11.set_ylabel("Pdyn (dots)")

axs[0].axvline(180, color='k', zorder=-1)
axs[1].axvline(180, color='k', zorder=-1)

plt.savefig("flight_sampled_a.png",dpi=300)
plt.savefig("flight_sampled_a.eps",dpi=300)
plt.savefig("flight_sampled_a.pdf",dpi=300)

plt.close()

import sys
# sys.exit()
import matplotlib.colors as colors
from matplotlib import ticker

RE =6371e3

plt.figure(constrained_layout=True)
zlen = data['x'].shape[2]
plt.pcolor(data['x'][:,:,zlen//2]/RE,data['y'][:,:,zlen//2]/RE, pid[:,:,zlen//2]*1e9, norm=colors.SymLogNorm(linthresh=1e-11*1e9, vmin=-1e-10*1e9, vmax=1e-10*1e9), cmap="turbo")

cb = plt.colorbar(label=r"$\Pi : D_\mathrm{sim}$ [nPa s$^{-1}$]")
tick_locator = ticker.MaxNLocator(nbins=20)
cb.locator = tick_locator
cb.update_ticks()



plt.contour(data['x'][:,:,zlen//2]/RE,data['y'][:,:,zlen//2]/RE, pdyn[:,:,zlen//2],linewidths=1)
plt.xlabel(r"$X/R_\mathrm{E}$")
plt.ylabel(r"$Y/R_\mathrm{E}$")
plt.gca().set_aspect(1)
plt.gca().set_xlim((6e7/RE,8.5e7/RE))
plt.gca().set_ylim((-3e7/RE,1e7/RE))



for sc in [0,1,2,3,4,5,6]:
   plt.gca().plot(pts[sc::7,0]/RE,pts[sc::7,1]/RE, color='k',linewidth=0.5)


for ti in [180]:
   # pass
   ptt = np.argwhere(sampled_path[:,0]==ti)
   for i,pti in enumerate(ptt):
      if i in [0,1,2,3]:
         marker = '.'
      else:
         marker = 'x'
      marker = 'o'
      plt.gca().scatter(pts[pti,0]/RE,pts[pti,1]/RE, zorder=10, marker=marker, s=15, edgecolor='k')

      for scs in [[0,1],[0,2],[0,3],[1,2],[2,3],[0,4],[0,5],[0,6],[4,5],[5,6]]:
         print(ptt, scs[0], pts[ptt,0][:,0])
         ptx = pts[ptt,0].squeeze()
         pty = pts[ptt,1].squeeze()
         xs = np.array([ptx[scs[0]], ptx[scs[1]]])/RE
         ys = np.array([pty[scs[0]], pty[scs[1]]])/RE
         plt.gca().plot(xs,ys, color='k',linestyle='dotted',linewidth=0.3)




plt.savefig(f"const_pid_a.png", dpi=300, bbox_inches='tight')
plt.savefig(f"const_pid_a.eps", dpi=300, bbox_inches='tight')
plt.savefig(f"const_pid_a.pdf", dpi=300, bbox_inches='tight')
plt.close()
