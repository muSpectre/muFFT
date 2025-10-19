#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_fft_tests.py

@author Till Junge <till.junge@altermail.ch>

@date   17 Jan 2018

@brief  Compare µSpectre's fft implementations to numpy reference

Copyright © 2018 Till Junge

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

import numpy as np
import pytest
from mpi4py import MPI

from NuMPI.Testing.Subdivision import suggest_subdivisions

import muFFT, muGrid

assert muFFT.has_mpi

engines = (["fftwmpi", "pfft"] if muFFT.has_mpi else []) + (
    ["pocketfft", "fftw"] if MPI.COMM_WORLD.size == 1 else []
)


@pytest.mark.parametrize("engine_str", engines)
def test_forward_inverse_2d(engine_str):
    nb_grid_pts = [6, 4]

    left_ghosts = [1, 1]
    right_ghosts = [1, 1]
    try:
        engine = muFFT.FFT(
            nb_grid_pts,
            engine=engine_str,
            communicator=MPI.COMM_WORLD,
            nb_ghosts_left=left_ghosts,
            nb_ghosts_right=right_ghosts,
        )
        engine.create_plan(1)
    except muFFT.UnknownFFTEngineError:
        # This FFT engine has not been compiled into the code. Skip
        # test.
        return

    field = engine.real_space_field("field")
    fourier_field = engine.fourier_space_field("fourier_field")
    result_field = engine.real_space_field("result_field_real")
    field.sg[...] = np.random.random(field.sg.shape)

    engine.fft(field, fourier_field)
    engine.ifft(fourier_field, result_field)
    result_field.sg *= engine.normalisation

    np.testing.assert_allclose(field.s, result_field.s, err_msg=f"Failed for engine {engine_str}")
