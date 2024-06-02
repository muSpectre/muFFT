import numpy as np
from muFFT import FFT

# Instantiate a FFT object with the PocketFFT engine
nb_grid_pts = (32, 32)
fft = FFT(nb_grid_pts)

# Obtain a real field and fill it
rfield = fft.real_space_field('scalar field')
rfield.p = np.random.rand(*nb_grid_pts)

# Compute Fourier transform
ffield = fft.fourier_space_field('scalar field')
fft.fft(rfield, ffield)

# Compute Fourier gradient by multiplying with wavevector
fgrad = fft.fourier_space_field('gradient field', (2,))
fgrad.p = fft.fftfreq * ffield.p

# Inverse transform to get gradient in real space
rgrad = fft.real_space_field('gradient field', (2,))
fft.ifft(fgrad, rgrad)
