import numpy as np
from mpi4py import MPI
from muGrid import FileIONetCDF, OpenMode, Communicator
from muFFT import FFT

# Instantiate a FFT object with the PocketFFT engine
nb_grid_pts = (32, 32, 32)
physical_sizes = (2, 2, 2)  # Sizes of the domain (in arbitrary units)
fft = FFT(nb_grid_pts, engine='mpi', communicator=MPI.COMM_WORLD)

if MPI.COMM_WORLD.rank == 0:
    print('  Rank   Size          Domain       Subdomain        Location')
    print('  ----   ----          ------       ---------        --------')
MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first

print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(fft.nb_domain_grid_pts):>15} '
      f'{str(fft.nb_subdomain_grid_pts):>15} {str(fft.subdomain_locations):>15}')

# Compute wavevectors (2 * pi * k / L for all k and in all directions)
wavevectors = (2 * np.pi * fft.ifftfreq.T / np.array(physical_sizes)).T

# Obtain a real field and fill it
rfield = fft.real_space_field('scalar-field')
x, y, z = fft.coords
rfield.p = np.sin(2 * np.pi * x + 4 * np.pi * y)  # Just a sine

# Compute Fourier transform
ffield = fft.fourier_space_field('scalar-field')
fft.fft(rfield, ffield)

# Compute Fourier gradient by multiplying with wavevector
fgrad = fft.fourier_space_field('gradient-field', (3,))
fgrad.p = 1j * wavevectors * ffield.p

# Inverse transform to get gradient in real space
rgrad = fft.real_space_field('gradient-field', (3,))
fft.ifft(fgrad, rgrad)

# Normalize gradient (µFFT does not normalize the transform)
gradx, grady, gradz = rgrad.p * fft.normalisation

# Gradient in x is cosine
lx, ly, lz = physical_sizes
np.testing.assert_allclose(
    gradx, 2 * np.pi * np.cos(2 * np.pi * x + 4 * np.pi * y) / lx, atol=1e-12)
# Gradient in y is also cosine
np.testing.assert_allclose(
    grady, 4 * np.pi * np.cos(2 * np.pi * x + 4 * np.pi * y) / ly, atol=1e-12)
# Gradient in z is zero
np.testing.assert_allclose(gradz, 0, atol=1e-12)

# I/O example
file = FileIONetCDF('example.nc', open_mode=OpenMode.Overwrite,
                    communicator=Communicator(MPI.COMM_WORLD))
file.register_field_collection(fft.real_field_collection)
file.append_frame().write()
