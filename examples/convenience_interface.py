import numpy as np
from muFFT import FFT

# Instantiate a FFT object with the PocketFFT engine
nb_grid_pts = (32, 32)
fft = FFT(nb_grid_pts, engine='pocketfft',
          allow_temporary_buffer=True)

# Create a random real-space field as a numpy array
rarr = np.random.rand(*nb_grid_pts)

# The convenience interface can work with numpy array directly, but this
# implies intermediate temporary copies of the numpy arrays
farr = fft.fft(rarr)

# Convert back
r2arr = fft.ifft(farr)

# Check that the original and the reconstructed field are the same
np.testing.assert_allclose(rarr, r2arr * fft.normalisation)
