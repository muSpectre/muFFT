import numpy as np
from muFFT import FFT
from muFFT.Stencils2D import upwind_x, upwind_y

# Instantiate a FFT object with the PocketFFT engine
nb_grid_pts = (54, 17)
physical_sizes = (1.4, 2.3)  # Sizes of the domain (in arbitrary units)
grid_spacing_x, grid_spacing_y = np.array(physical_sizes) / np.array(nb_grid_pts)
fft = FFT(nb_grid_pts)

# Obtain a real field and fill it
rfield = fft.real_space_field('scalar field')
x, y = fft.coords
rfield.p = np.sin(2 * np.pi * x)  # Just a sine

# Compute Fourier transform
ffield = fft.fourier_space_field('scalar field')
fft.fft(rfield, ffield)

# Compute gradient by multiplying with wavevector
fgrad = fft.fourier_space_field('gradient field', (2,))
fgrad.p[0] = upwind_x.fourier(fft.fftfreq) * ffield.p / grid_spacing_x
fgrad.p[1] = upwind_y.fourier(fft.fftfreq) * ffield.p / grid_spacing_y

# Inverse transform to get gradient in real space
rgrad = fft.real_space_field('gradient field', (2,))
fft.ifft(fgrad, rgrad)

# Normalize gradient (µFFT does not normalize the transform)
gradx, grady = rgrad.p * fft.normalisation

# Gradient in x is the finite difference of the sine
lx, ly = physical_sizes
np.testing.assert_allclose(gradx, (np.roll(rfield.p, -1, 0) - rfield.p) / grid_spacing_x, atol=1e-12)
# Gradient in y is zero
np.testing.assert_allclose(grady, 0, atol=1e-12)
