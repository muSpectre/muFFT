Python Bindings
###############

Fast-Fourier Transform
**********************

The core of *µ*\FFT is the Fast-Fourier-Transform (FFT) abstraction layer,
implemented as a class which is specialized for a given FFT engine. Currently
supported are
`pocketfft <https://github.com/mreineck/pocketfft>`_,
`FFTW <https://www.fftw.org>`_,
`MPIFFTW <https://www.fftw.org/fftw3_doc/FFTW-MPI-Installation.html>`_
and
`PFFT <https://github.com/mpip/pfft>`_.
In our experience, *PFFT* has the best scaling properties for large-scale parallel
FFTs because it uses pencil decomposition. Only *pocketfft* is enabled by default.

The FFT class
*************

Instantiating an FFT class is a simple as

.. code-block:: python

    from muFFT import FFT
    fft = FFT([nx, ny, nz], engine='pocketfft')

where `[nx, ny, nz]` is the shape of the grid and the optional `engine` is the FFT
engine to use. The FFT class takes another optional `comm` parameter that can be
either a `mpi4py` communicator or an native muFFT `Communicator`. If no communicator
is provided, the FFT class will use the default communicator `MPI_COMM_NULL`.

Convenience interface
*********************

*µ*\Grid interface
******************

