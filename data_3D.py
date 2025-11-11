import numpy as np
import matplotlib.pyplot as plt

R1Y_data = np.load('data/data/3D_data/R1Y_3D.npz')

print(R1Y_data.keys())

vtot = R1Y_data['vtot']

print(vtot.shape)

# the shape of the array is NZ,NY,NX
# let's plot a central slice of the data in the Z direction

plt.imshow(vtot[200,:,:],origin="lower")

plt.savefig("vtot_xy_plane_test.png", dpi=300)