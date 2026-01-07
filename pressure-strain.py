# Example script to produce and plot pressure-strain interaction terms from Amitis simulation data
# Assumes the presence of a gridded .npz file with velocity and pressure tensor components


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


data = np.load("data/Gridded_v_p_093000_improvPStats.npz")
data = {k:v for k,v in data.items()}
ks = list(data.keys())
print(ks)
xlen = data['z'].shape[0]
ylen = data['z'].shape[1]
zlen = data['z'].shape[2]

kernel_len = 3
conv_order = 1
kernel = np.ones([kernel_len, kernel_len,kernel_len])
kernel = kernel/np.sum(kernel)

for var in ks[4:]:
    for i in range(conv_order):
        data[var] = sp.signal.convolve(data[var], kernel, mode="same", method="direct")
    plt.figure(constrained_layout=True)

    plt.pcolor(data['x'][:,:,zlen//2],data['y'][:,:,zlen//2], data[var][:,:,zlen//2])
    plt.gca().set_aspect('equal', 'box')
    plt.colorbar(label=f"{var}")
    plt.xlabel("x/m")
    plt.ylabel("y/m")

    plt.savefig(f"{var}_xy.png", dpi=300, bbox_inches='tight')
    plt.close()

# for var in ks[4:]:
#     for i in range(conv_order):
#         data[var] = sp.signal.convolve(data[var], kernel, mode="same", method="direct")
#     plt.pcolor(data['x'][:,ylen//2,:],data['z'][:,ylen//2,:], data[var][:,ylen//2,:])
#     plt.gca().set_aspect('equal', 'box')
#     plt.xlabel("x/m")
#     plt.ylabel("z/m")
#     plt.colorbar(label=f"{var}")

#     plt.savefig(f"{var}_xz.png", dpi=300, bbox_inches='tight')
#     plt.close()

# v = np.stack((data["vx"],data["vy"],data["vz"]),axis=-1)
ptensor = np.stack((data["p_xx"],data["p_xy"],data["p_xz"],
                    data["p_xy"],data["p_yy"],data["p_yz"],
                    data["p_xz"],data["p_yz"],data["p_zz"]),axis=-1)

# print(v.shape)

ptensor = np.reshape(ptensor, [xlen,ylen,zlen,3,3])
# print(ptensor.shape)
plt.figure(constrained_layout=True)
plt.pcolor(data['x'][:,:,zlen//2],data['y'][:,:,zlen//2], np.linalg.norm(ptensor[:,:,zlen//2,:,:],ord="nuc",axis=(2,3)))
plt.gca().set_aspect('equal', 'box')

plt.colorbar(label=f"ptensor nuc")
plt.xlabel("x/m")
plt.ylabel("y/m")

plt.savefig(f"ptensor.png", dpi=300, bbox_inches='tight')
plt.close()

# for i in [0,1,2]:
#  for j in [0,1,2]:
#     plt.pcolor(data['x'][:,:,zlen//2],data['y'][:,:,zlen//2],ptensor[:,:,zlen//2,i,j])
#     plt.axis("equal")
#     plt.colorbar(label=f"ptensor nuc")

#     plt.savefig(f"ptensor{i}{j}.png", dpi=300)
#     plt.close()

v_jacobian = np.stack((*np.gradient(data["vx"], 1e6),*np.gradient(data["vy"], 1e6),*np.gradient(data["vz"], 1e6)), axis = -1)
v_jacobian = np.reshape(v_jacobian, [xlen,ylen,zlen, 3,3])
# print(v_jacobian.shape)
# for i in [0,1,2]:
#  for j in [0,1,2]:
#     plt.pcolor(data['x'][:,:,zlen//2],data['y'][:,:,zlen//2],v_jacobian[:,:,zlen//2,i,j])
#     plt.axis("equal")
#     plt.colorbar(label=f"vjac{i}{j}")

#     plt.savefig(f"vjac{i}{j}.png", dpi=300)
#     plt.close()


# https://github.com/fmihpc/analysator/blob/c6a78122e30410f62cb7d7e2406dd0531f722e71/analysator/pyVlsv/reduction.py#L960
def PiD(v_jacobian, ptensor):
   ''' Calculate the (proton) Pi-D interaction term
       See e.g. Yang+2017: https://doi.org/10.1103/PhysRevE.95.061201
   '''

   kdelta = np.einsum('i,jk',np.ones(len(ptensor)),np.eye(3))  # Identity tensors in the shape of ptensor
   p = 1/3*np.einsum('...ii', ptensor)   # Scalar pressure

   pi = ptensor - np.einsum('i..., i...->i...', p, kdelta)  # Traceless pressure tensor

   div_v = np.einsum('...ii', v_jacobian)    # Velocity divergence

   d = 0.5*(v_jacobian+np.einsum('...ji', v_jacobian)) - 1/3 * np.einsum('i..., i...->i...', div_v, kdelta)   # D tensor
   pi_d = -np.einsum('...ij,...ij', pi, d)
   return pi_d

# https://github.com/fmihpc/analysator/blob/c6a78122e30410f62cb7d7e2406dd0531f722e71/analysator/pyVlsv/reduction.py#L992
def Pressure_strain(v_jacobian, ptensor):
   ''' Calculate the (proton) pressure strain interaction -(P dot nabla) dot bulk velocity, a sum of the pdil and PiD interactions. A separate datareducer to avoid redundant Jacobian estimations.
       See e.g. Yang+2017: https://doi.org/10.1103/PhysRevE.95.061201
   '''


   kdelta = np.einsum('i,jk',np.ones(len(ptensor)),np.eye(3))
   p = 1/3*np.einsum('...ii', ptensor)

   pi = ptensor - np.einsum('i..., i...->i...', p, kdelta)

   div_v = np.einsum('...ii', v_jacobian)

   d = 0.5*(v_jacobian+np.einsum('...ji', v_jacobian)) - 1/3 * np.einsum('i..., i...->i...', div_v, kdelta)   

   strain = -np.einsum('...ij,...ij', pi, d) -1/3*np.einsum('...ii', ptensor)*np.einsum('...ii', v_jacobian)

   return strain

# https://github.com/fmihpc/analysator/blob/c6a78122e30410f62cb7d7e2406dd0531f722e71/analysator/pyVlsv/reduction.py#L938
def Pressure_dilatation(v_jacobian, p):
   ''' Calculate the (proton) pressure dilatation interaction term -p*div(V)
       See e.g. Yang+2017: https://doi.org/10.1103/PhysRevE.95.061201
   '''

   div_v = np.einsum('...ii', v_jacobian)
   
   dilatation = -p*div_v

   return dilatation

pidresult = PiD(v_jacobian.reshape((xlen*ylen*zlen,3,3)), ptensor.reshape((xlen*ylen*zlen,3,3))).reshape((xlen,ylen,zlen))
strainresult = Pressure_strain(v_jacobian.reshape((xlen*ylen*zlen,3,3)), ptensor.reshape((xlen*ylen*zlen,3,3))).reshape((xlen,ylen,zlen))
dilatationresult = Pressure_dilatation(v_jacobian.reshape((xlen*ylen*zlen,3,3)), 1/3.*np.trace(ptensor.reshape((xlen*ylen*zlen,3,3)), axis1=1,axis2=2)).reshape((xlen,ylen,zlen))

import matplotlib.colors as colors
from matplotlib import ticker

plt.figure(constrained_layout=True)
plt.pcolor(data['x'][:,:,zlen//2],data['y'][:,:,zlen//2], pidresult[:,:,zlen//2], norm=colors.SymLogNorm(linthresh=1e-11, vmin=-1e-10, vmax=1e-10), cmap="turbo")

cb = plt.colorbar(label="pid (linear map until $\pm10^{-11}$)")
tick_locator = ticker.MaxNLocator(nbins=20)
cb.locator = tick_locator
cb.update_ticks()
plt.contour(data['x'][:,:,zlen//2],data['y'][:,:,zlen//2], data['pdyn'][:,:,zlen//2],linewidths=1)
plt.xlabel("x/m")
plt.ylabel("y/m")
plt.gca().set_aspect('equal', 'box')
plt.savefig(f"pid.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(constrained_layout=True)
plt.pcolor(data['x'][:,:,zlen//2],data['y'][:,:,zlen//2], strainresult[:,:,zlen//2], norm=colors.SymLogNorm(linthresh=1e-11, vmin=-1e-10, vmax=1e-10), cmap="turbo")
cb = plt.colorbar(label="pressure-strain (linear map until $\pm10^{-11}$)")
tick_locator = ticker.MaxNLocator(nbins=20)
cb.locator = tick_locator
cb.update_ticks()
plt.contour(data['x'][:,:,zlen//2],data['y'][:,:,zlen//2], data['pdyn'][:,:,zlen//2],linewidths=1)
plt.xlabel("x/m")
plt.ylabel("y/m")
plt.gca().set_aspect('equal', 'box')
plt.savefig(f"pressure-strain.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(constrained_layout=True)
plt.pcolor(data['x'][:,:,zlen//2],data['y'][:,:,zlen//2], dilatationresult[:,:,zlen//2], norm=colors.SymLogNorm(linthresh=1e-11, vmin=-1e-10, vmax=1e-10), cmap="turbo")
cb = plt.colorbar(label="pressure-dilatation (linear map until $\pm10^{-11}$)")
tick_locator = ticker.MaxNLocator(nbins=20)
cb.locator = tick_locator
cb.update_ticks()
plt.contour(data['x'][:,:,zlen//2],data['y'][:,:,zlen//2], data['pdyn'][:,:,zlen//2],linewidths=1)
plt.xlabel("x/m")
plt.ylabel("y/m")
plt.gca().set_aspect('equal', 'box')
plt.savefig(f"pressure-dilatation.png", dpi=300, bbox_inches='tight')
plt.close()