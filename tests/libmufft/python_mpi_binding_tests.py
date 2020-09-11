#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_mpi_binding_tests.py

@author Till Junge <till.junge@epfl.ch>

@date   28 Feb 2019

@brief  Unit tests for python bindings with MPI support

Copyright © 2019 Till Junge

µFFT is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µFFT is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µFFT; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import unittest
import numpy as np

from python_test_imports import muFFT

from python_fft_tests import FFT_Check
#from python_netcdf_tests import NetCDF_Check_2d, NetCDF_Check_3d

if __name__ == '__main__':
    unittest.main()
