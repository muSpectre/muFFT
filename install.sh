#! /bin/bash

# Install MPI-parallel muFFT and all dependencies from source

# Assumes that MPI is present

# Installation prefix
PREFIX=${HOME}/.local

# Versions to install
PNETCDF_VERSION=1.13.0
FFTW_VERSION=3.3.10

# Download and compile in /tmp
WORKDIR=/tmp

#
# Install parallel version of the NetCDF library from the sources.
# This is necessary because parallel compiles (if existing) are
# broken on most distributions.
#
curl https://parallel-netcdf.github.io/Release/pnetcdf-${PNETCDF_VERSION}.tar.gz | tar -xz -C ${WORKDIR}
cd ${WORKDIR}/pnetcdf-${PNETCDF_VERSION}
./configure --disable-shared --enable-static --with-pic --disable-fortran --disable-cxx --prefix=${PREFIX}
make -j 4
make install

#
# Install FFTW3 with MPI support
#
curl -L http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz | tar -xz -C ${WORKDIR}
cd ${WORKDIR}/fftw-${FFTW_VERSION}
./configure --disable-shared --enable-static --with-pic --enable-mpi --disable-fortran --enable-sse2 --enable-avx --enable-avx2 --prefix=${PREFIX}
make -j 4
make install

#
# Install current master of PFFT
#
git clone https://github.com/mpip/pfft.git ${WORKDIR}/pfft
cd ${WORKDIR}/pfft
./bootstrap.sh
CFLAGS="-O3" ./configure --disable-shared --enable-static --with-pic --disable-fortran --prefix=${PREFIX}
make -j 4
make install

