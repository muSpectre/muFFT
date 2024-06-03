#! /bin/bash

# Install MPI-parallel muFFT and all dependencies from source

# Assumes that MPI is present

# Installation prefix
if [ -z ${PREFIX} ]; then
    PREFIX=${HOME}/.local
fi

# Versions to install
PNETCDF_VERSION=1.13.0
FFTW_VERSION=3.3.10

# Download and compile in /tmp
if [ -z ${WORKDIR} ]; then
    WORKDIR=/tmp
fi

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
./configure --disable-shared --enable-static --with-pic --disable-fortran --enable-mpi --prefix=${PREFIX}
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
