import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spint

inputfile = "data/Gridded_v_p_093000_improvPStats_withPiD.npz"

data = np.load(inputfile)

data = {k:v for k,v in data.items()}
print(data.keys())
vx = data['vx']
pdyn = data['pdyn']


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
print(x)
print(x.shape, y.shape, z.shape)
print(vx.shape)

# along vlasiator-data-interface/example_flight.py

flight = np.loadtxt("trajectories/PO_constellation_flight.txt", delimiter=',')

pts = flight[:,2:5] # convert to km from m
t = flight[::7,0]

print(pts)
intp = spint.RegularGridInterpolator( (x,y,z), vx, bounds_error=False, fill_value=np.nan)
vals = intp(pts)

fig, axs = plt.subplots(2)

#plot outer tetra vtot
for po in [0,1,2,3]:
   axs[0].plot(t, vals[po::7], label="outer:{:d}".format(po))

#plot inner tetra vtot
for po in [0,4,5,6]:
   axs[1].plot(t, vals[po::7], label="inner:{:d}".format(po))

axs[0].set_ylabel("vx (line)")
axs[1].set_ylabel("vx (line)")
axs[0].legend()
axs[1].legend()

intp = spint.RegularGridInterpolator( (x,y,z), pdyn, bounds_error=False, fill_value=np.nan)
vals = intp(pts)

axs01 = axs[0].twinx()
axs11 = axs[1].twinx()

#plot outer tetra pdyn
for po in [0,1,2,3]:
   axs01.plot(t, vals[po::7], label="outer:{:d}".format(po),linestyle='dotted')

#plot inner tetra pdyn
for po in [0,4,5,6]:
   axs11.plot(t, vals[po::7], label="inner:{:d}".format(po),linestyle='dotted')

axs01.set_ylabel("Pdyn (dots)")
axs11.set_ylabel("Pdyn (dots)")



plt.savefig("flight_test.png",dpi=300)