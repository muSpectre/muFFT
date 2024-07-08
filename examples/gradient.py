import numpy as np
from muFFT import FFT

# Instantiate a FFT object with the PocketFFT engine
nb_grid_pts = (54, 17)
physical_sizes = (1.4, 2.3)  # Sizes of the domain (in arbitrary units)
fft = FFT(nb_grid_pts)

# Compute wavevectors (2 * pi * k / L for all k and in all directions)
wavevectors = (2 * np.pi * fft.ifftfreq.T / np.array(physical_sizes)).T

# Obtain a real field and fill it
rfield = fft.real_space_field('scalar field')
x, y = fft.coords
rfield.p = np.sin(2 * np.pi * x)  # Just a sine

# Compute Fourier transform
ffield = fft.fourier_space_field('scalar field')
fft.fft(rfield, ffield)

# Compute Fourier gradient by multiplying with wavevector
fgrad = fft.fourier_space_field('gradient field', (2,))
fgrad.p = 1j * wavevectors * ffield.p

# Inverse transform to get gradient in real space
rgrad = fft.real_space_field('gradient field', (2,))
fft.ifft(fgrad, rgrad)

# Normalize gradient (µFFT does not normalize the transform)
gradx, grady = rgrad.p * fft.normalisation

# Gradient in x is cosine
lx, ly = physical_sizes
np.testing.assert_allclose(gradx, 2 * np.pi * np.cos(2 * np.pi * x) / lx, atol=1e-12)
# Gradient in y is zero
np.testing.assert_allclose(grady, 0, atol=1e-12)
