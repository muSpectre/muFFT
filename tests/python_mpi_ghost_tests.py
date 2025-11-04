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

from NuMPI.Testing.Subdivision import suggest_subdivisions

import muFFT, muGrid

if muFFT.has_mpi:
    from mpi4py import MPI

    communicator = muFFT.Communicator(MPI.COMM_WORLD)
else:
    communicator = muFFT.Communicator()

engines = (["fftwmpi", "pfft"] if muFFT.has_mpi else []) + (
    ["pocketfft", "fftw"] if communicator.size == 1 else []
)


@pytest.mark.skipif(
    communicator.size > 4,
    reason="Tests only for up to 4 MPI processes; empty process domains may occur otherwise.",
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
            communicator=communicator,
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
    field.s[...] = np.random.random(field.s.shape)

    engine.fft(field, fourier_field)
    engine.ifft(fourier_field, result_field)
    result_field.s *= engine.normalisation

    np.testing.assert_allclose(
        field.s, result_field.s, err_msg=f"Failed for engine {engine_str}"
    )


@pytest.mark.skipif(
    communicator.size > 4,
    reason="Tests only for up to 4 MPI processes; empty process domains may occur otherwise.",
)
@pytest.mark.parametrize("engine_str", engines)
def test_forward_inverse_3d(engine_str):
    nb_grid_pts = [6, 4, 4]

    left_ghosts = [1, 1, 1]
    right_ghosts = [1, 1, 1]
    try:
        engine = muFFT.FFT(
            nb_grid_pts,
            engine=engine_str,
            communicator=communicator,
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
    field.s[...] = np.random.random(field.s.shape)

    engine.fft(field, fourier_field)
    engine.ifft(fourier_field, result_field)
    result_field.s *= engine.normalisation

    np.testing.assert_allclose(
        field.s, result_field.s, err_msg=f"Failed for engine {engine_str}"
    )


@pytest.mark.skipif(
    communicator.size > 4,
    reason="Tests only for up to 4 MPI processes; empty process domains may occur otherwise.",
)
@pytest.mark.parametrize("engine_str", engines)
def test_apply_stencil(engine_str):
    # Two dimensional grid
    nx, ny = nb_grid_pts = [1024, 10]

    left_ghosts = [1, 1]
    right_ghosts = [1, 1]

    try:
        engine = muFFT.FFT(
            nb_grid_pts,
            engine=engine_str,
            communicator=communicator,
            nb_ghosts_left=left_ghosts,
            nb_ghosts_right=right_ghosts,
        )
        engine.create_plan(1)
    except muFFT.UnknownFFTEngineError:
        # This FFT engine has not been compiled into the code. Skip
        # test.
        return

    fc = engine.real_field_collection
    fc.set_nb_sub_pts("quad_points", 2)
    fc.set_nb_sub_pts("nodal_points", 1)

    # Get nodal field
    nodal_field = fc.real_field("nodal-field", (1,), "nodal_points")

    # Get quadrature field of shape (2, quad, nx, ny)
    quad_field = fc.real_field("quad-field", (2,), "quad_points")

    # Fill nodal field with a sine-wave
    x, y = nodal_field.icoords
    nodal_field.p[0] = np.sin(2 * np.pi * x / nx)

    # Derivative stencil of shape (2, quad, 2, 2)
    gradient = np.array(
        [
            [  # Derivative in x-direction
                [[[-1, 0], [1, 0]]],  # Bottom-left triangle (first quadrature point)
                [[[0, -1], [0, 1]]],  # Top-right triangle (second quadrature point)
            ],
            [  # Derivative in y-direction
                [[[-1, 1], [0, 0]]],  # Bottom-left triangle (first quadrature point)
                [[[0, 0], [-1, 1]]],  # Top-right triangle (second quadrature point)
            ],
        ],
    )
    op = muGrid.ConvolutionOperator([0, 0], gradient)

    # Communicate ghosts
    fft.communicate_ghosts(nodal_field)

    # Apply the gradient operator to the nodal field and write result to the quad field
    op.apply(nodal_field, quad_field)

    # Check that the quadrature field has the correct derivative
    np.testing.assert_allclose(
        quad_field.s[0, 0],
        2 * np.pi * np.cos(2 * np.pi * (x + 0.25) / nx) / nx,
        atol=1e-5,
    )


@pytest.mark.parametrize("engine_str", engines)
def test_unit_impuls(engine_str):
    # Two dimensional grid
    nx, ny = nb_grid_pts = [4, 6]

    left_ghosts = [1, 1]
    right_ghosts = [1, 1]

    try:
        engine = muFFT.FFT(
            nb_grid_pts,
            engine=engine_str,
            communicator=communicator,
            nb_ghosts_left=left_ghosts,
            nb_ghosts_right=right_ghosts,
        )
        engine.create_plan(1)
    except muFFT.UnknownFFTEngineError:
        # This FFT engine has not been compiled into the code. Skip
        # test.
        return

    fc = engine.real_field_collection
    fc.set_nb_sub_pts("quad_points", 2)
    fc.set_nb_sub_pts("nodal_points", 1)

    # Get nodal field
    nodal_field = fc.real_field("nodal-field", (1,), "nodal_points")
    impuls_response_field = fc.real_field("impuls_response_field", (1,), "nodal_points")

    # Get quadrature field of shape (2, quad, nx, ny)
    quad_field = fc.real_field("quad-field", (2,), "quad_points")

    impuls_locations = (impuls_response_field.icoordsg[0] == 0) & (impuls_response_field.icoordsg[1] == 0)
    left_location = (impuls_response_field.icoordsg[0] == nx - 1) & (impuls_response_field.icoordsg[1] == 0)
    right_location = (impuls_response_field.icoordsg[0] == 1) & (impuls_response_field.icoordsg[1] == 0)

    top_location = (impuls_response_field.icoordsg[0] == 0) & (impuls_response_field.icoordsg[1] == 1)
    bottom_location = (impuls_response_field.icoordsg[0] == 0) & (impuls_response_field.icoordsg[1] == ny - 1)

    nodal_field.sg[0, 0, impuls_locations] = 1
    impuls_response_field.sg[0, 0, impuls_locations] = 4
    impuls_response_field.sg[0, 0, left_location] = -1
    impuls_response_field.sg[0, 0, right_location] = -1
    impuls_response_field.sg[0, 0, top_location] = -1
    impuls_response_field.sg[0, 0, bottom_location] = -1

    print(f'unit impuls: nodal field with buffers in rank {MPI.COMM_WORLD.rank} \n ' + f'{nodal_field.s}')
    print(
        f'impuls_response_field: nodal field with buffers in rank {MPI.COMM_WORLD.rank} \n ' + f'{impuls_response_field.s}')

    engine.communicate_ghosts(nodal_field)
    # Derivative stencil of shape (2, quad, 2, 2)
    gradient = np.array(
        [
            [  # Derivative in x-direction
                [[[-1, 0], [1, 0]]],  # Bottom-left triangle (first quadrature point)
                [[[0, -1], [0, 1]]],  # Top-right triangle (second quadrature point)
            ],
            [  # Derivative in y-direction
                [[[-1, 1], [0, 0]]],  # Bottom-left triangle (first quadrature point)
                [[[0, 0], [-1, 1]]],  # Top-right triangle (second quadrature point)
            ],
        ],
    )
    gradient_op = muGrid.ConvolutionOperator([0, 0], gradient)

    # Apply the gradient operator to the nodal field and write result to the quad field
    gradient_op.apply(nodal_field, quad_field)

    engine.communicate_ghosts(quad_field)

    # Apply the gradient transposed operator to the quad field and write result to the nodal field
    gradient_op.transpose(quadrature_point_field=quad_field,
                          nodal_field=nodal_field,
                          weights=[1 / 2, 1 / 2]  # size of the element is half of the pixel. Pixel size is 1
                          )

    print(
        f'computed unit impuls response : nodal field with buffers in rank {MPI.COMM_WORLD.rank} \n ' + f'{nodal_field.s}')

    print(f'local sum on core (nodal_field.s) = {np.sum(nodal_field.s)}')  # does not have to be zero
    local_sum = np.sum(nodal_field.s)
    total_sum = communicator.sum(local_sum)
    print(f'total_sum = {total_sum}')  # have to be zero

    # Check that the nodal_field has zero mean
    np.testing.assert_allclose(
        total_sum, 0,
        atol=1e-10, )
    # Check that the impulse response is correct
    np.testing.assert_allclose(
        nodal_field.s, impuls_response_field.s,
        atol=1e-5, )
