import numpy as np
from muFFT import FFT

# Instantiate a FFT object with the PocketFFT engine
nb_grid_pts = (54, 17)
physical_sizes = (1.4, 2.3)  # Sizes of the domain (in arbitrary units)
nx, ny = nb_grid_pts
lx, ly = physical_sizes
fft = FFT(nb_grid_pts)

# Obtain a real field and fill it
rfield = fft.real_space_field('scalar field')
x, y = fft.coords
rfield.p = np.sin(2 * np.pi * x)  # Just a sine

# Compute Fourier transform
ffield = fft.fourier_space_field('scalar field')
fft.fft(rfield, ffield)

# Compute Fourier gradient by multiplying with wavevector
fgrad = fft.fourier_space_field('gradient field', (2,))
fgrad.p = 2 * np.pi * 1j * fft.fftfreq * ffield.p

# Inverse transform to get gradient in real space
rgrad = fft.real_space_field('gradient field', (2,))
fft.ifft(fgrad, rgrad)

# Normalize gradient
gradx, grady = rgrad.p * fft.normalisation
gradx *= nx / lx  # Need to multiply with inverse grid spacing
grady *= ny / ly  # Need to multiply with inverse grid spacing

# Gradient in x is cosine
np.testing.assert_allclose(gradx, 2 * np.pi * np.cos(2 * np.pi * x) / lx, atol=1e-12)
# Gradient in y is zero
np.testing.assert_allclose(grady, 0, atol=1e-12)
