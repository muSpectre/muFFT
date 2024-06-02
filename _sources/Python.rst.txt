Python Bindings
###############

Fast-Fourier Transform
**********************

The core of µFFT is the Fast-Fourier-Transform (FFT) abstraction layer,
implemented as a class which is specialized for a given FFT engine. Currently
supported are
`pocketfft <https://github.com/mreineck/pocketfft>`_,
`FFTW <https://www.fftw.org>`_,
`MPIFFTW <https://www.fftw.org/fftw3_doc/FFTW-MPI-Installation.html>`_
and
`PFFT <https://github.com/mpip/pfft>`_.
In our experience, *PFFT* has the best scaling properties for large-scale parallel
FFTs because it uses pencil decomposition. Only *pocketfft* is enabled by default.

The FFTs operate on `µGrid <https://github.com/muSpectre/muGrid>`_ fields. The
engines creates fields with the right memory layout for the respective FFT engine.
Please read the µGrid
`documentation <https://muspectre.github.io/muGrid/Python.html>`_ before starting here.

The FFT class
*************

Instantiating an FFT class is a simple as

.. code-block:: python

    from muFFT import FFT
    fft = FFT((nx, ny, nz), engine='pocketfft')

where `[nx, ny, nz]` is the shape of the grid and the optional `engine` is the FFT
engine to use. The FFT class takes another optional `comm` parameter that can be
either a `mpi4py` communicator or an native muFFT `Communicator`. If no communicator
is provided, the FFT class will use the default communicator `MPI_COMM_SELF`.

µGrid interface
***************

The FFT class provides methods to obtain real-valued real-space fields and complex-valued
Fourier-space fields. The transform operates between these fields. An example of a simple
transform is shown here:

.. literalinclude:: ../../examples/transform.py
    :language: python
