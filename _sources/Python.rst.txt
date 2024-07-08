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

Instantiating an FFT object is a simple as

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

Convenience numpy interface
***************************

The FFT class also provides a convenience interface that allows usage of `numpy` arrays
directly. Internally, the `numpy` arrays are converted to µGrid fields. The following
shows an example of the convenience interface:

.. literalinclude:: ../../examples/convenience_interface.py
    :language: python

The downside of the convenience interface is that temporary copies are typically created.
The reason for this is that most FFT engines have strict requirements on the memory
layout. The µGrid interface allows to create fields with the right memory layout directly,
avoiding copies.

Temporary copies can be disabled entirely by setting the `allow_temporary_buffer` option
to `false`. The above example then still runs for the `pocketfft` engine, but will fail
with the error::

    RuntimeError: Incompatible memory layout for the real-space field and no temporary
    copies are allowed.

for `fftw` which requires a padded memory layout.

Normalization
*************

All µFFT transforms are not normalized. A round-trip forward-inverse transform will pick
up a global factor, that can be compensated by multiplying with the value of the
`normlization` property of the FFT object as in the examples above.

The reason for this is that there is a choice of putting the factor into the forward or
inverse transform, or putting the square-root of the factor into both. µFFT leaves the
choice of normalization to the user.

More information can be found for example in the
`section on normalization <https://numpy.org/doc/stable/reference/routines.fft.html#normalization>`_
of the `numpy` documentation.

Coordinates and wavevectors
***************************

Wavevectors are obtained via the `fftfreq` property of the FFT object. The result is
identical to the
`numpy.fft.fftfreq <https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html>`_
function. µFFT adds a second convenience method `ifftfreq`, which returns an array of integers
that indicate the index of the wavevector in the Fourier-space field. This is the same as the output
of `numpy.fft.fftfreq <https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html>`_
with `d=1/n`. The following example shows how to obtain a gradient field using a Fourier derivative
that uses the `ifftfreq` property:

.. literalinclude:: ../../examples/gradient.py
    :language: python

The result of `fftfreq` can be regarded as a fractional Fourier-space coordinate,
the result of `ifftfreq` and an integer Fourier-space coordinate.
There are also convenience properties `coords` and `icoords` that yield the fractional
and integer coordinates in real-space. These properties are useful in particular when
running MPI-parallel with domain decomposition. All properties then return just the coordinates of
the local domain.

Parallelization
***************

The previous example can be parallelized by initializing the FFT engine with an MPI communicator:

.. literalinclude:: ../../examples/parallel.py
    :language: python

The engine that is selected must support MPI parallelization. Currently, only the `fftwmpi` engine
and `pfft` engine support MPI parallelization. Using `mpi` as in the example autoselects an engine.

The parallelization employs domain decomposition. The domain is split into stripe-shaped subdomains
for `fftwmpi` and pencil-shaped subdomains for `pfft`. `pfft` scales better to large numbers of MPI
processes because of this pencil decomposition. The number of grid points on the local domain is
returned by `nb_subdomain_grid_pts` and the location of the domain is given by `subdomain_locations`.
Note that in a typical code uses `coords`, `icoord`, `fftfreq` and `ifftfreq` as above and does not
need to care about the actual details of the decomposition, as those properties return the domain-local
coordinates and wavevectors.

The above example also illustrates how to write the global field to a file.
