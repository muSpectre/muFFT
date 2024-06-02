import numpy as np
from muFFT import FFT

# Instantiate a FFT object with the PocketFFT engine
nb_grid_pts = (32, 32)
fft = FFT(nb_grid_pts, engine='pocketfft')

# Obtain a real and a Fourie-space field
rfield = fft.real_space_field('rfield')
ffield = fft.fourier_space_field('ffield')

# Fill field with random numbers
rfield.p = np.random.rand(*nb_grid_pts)

# Perform a forward FFT
fft.fft(rfield, ffield)

# Perform an inverse FFT
r2field = fft.real_space_field('second rfield')
fft.ifft(ffield, r2field)

# Check that the original and the reconstructed field are the same
np.testing.assert_allclose(rfield.p, r2field.p * fft.normalisation)
