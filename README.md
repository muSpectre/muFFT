# µFFT

µFFT is a unified interface to serial and MPI-parallel FFT libraries, build on
top of [µGrid](https://github.com/muSpectre/muGrid).
µGrid and µFFT make it easy to implement algorithms that operate on fields,
such as solving partial  differential equations. It supports parallelization
using domain decomposition  implemented using the Message Passing Interface (MPI).

µFFT is written in C++ and currently has language bindings for
[Python](https://www.python.org/).

This README contains only a small quick start guide. Please refer to the
[full documentation](https://muspectre.github.io/muFFT/) for more help.

## Quick start

To install µFFT, run

    pip install muFFT

Note that on most platforms this will install a binary wheel, that was
compiled with a minimal configuration. To compile for your specific platform
use

    pip install -v --no-binary muFFT muFFT

which will compile the code. µFFT will autodetect
µFFT will autodetect
[MPI](https://www.mpi-forum.org/),
[FFTW](https://www.fftw.org/),
[MPIFFTW](https://www.fftw.org/fftw3_doc/FFTW-MPI-Installation.html)
and
[PFFT](https://github.com/mpip/pfft).
Monitor output to see which of these options were automatically detected.

## Funding

This development has received funding from the
[Swiss National Science Foundation](https://www.snf.ch/en)
within an Ambizione Project and by the
[European Research Council](https://erc.europa.eu) within
[Starting Grant 757343](https://cordis.europa.eu/project/id/757343).
