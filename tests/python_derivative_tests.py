#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_derivative_tests.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   20 Jun 2019

@brief  test discrete derivative

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

import unittest
import numpy as np
import itertools

from python_test_imports import muFFT


class DerivativeCheck1d(unittest.TestCase):
    def setUp(self):
        self.nb_pts = [23]
        np.random.seed(7)
        self.field = np.empty(self.nb_pts, order='f')
        self.field[:] = np.random.random(self.nb_pts)
        self.fft = muFFT.FFT(self.nb_pts)
        self.nb_dof = 1
        self.fft.create_plan(self.nb_dof)
        self.fourier_field = self.fft.register_fourier_space_field(
            "fft_workspace", self.nb_dof)
        self.fft.fft(self.field, self.fourier_field)

    def test_upwind_differences(self):
        diffop = muFFT.Stencils1D.upwind
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            ndiff = self.field[(x+1) % nx] - self.field[x]
            ndiff = np.squeeze(ndiff)
            self.assertAlmostEqual(diff_field[x], ndiff)

    def test_central_differences(self):
        diffop = muFFT.Stencils1D.central
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            ndiff = (self.field[(x+1) % nx] - self.field[(x-1) % nx])/2
            ndiff = np.squeeze(ndiff)
            self.assertAlmostEqual(diff_field[x], ndiff)

    def test_central_2nd_differences(self):
        diffop = muFFT.Stencils1D.central_2nd
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            ndiff = self.field[(x+1) % nx] - 2*self.field[x] \
                + self.field[(x-1) % nx]
            ndiff = np.squeeze(ndiff)
            self.assertAlmostEqual(diff_field[x], ndiff)

    def test_sixth_order_central_differences(self):
        diffop = muFFT.Stencil1D.central6
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            ndiff = - 1/60*self.field[(x-3) % nx] \
                    + 3/20*self.field[(x-2) % nx] \
                    - 3/4*self.field[(x-1) % nx] \
                    + 3/4*self.field[(x+1) % nx] \
                    - 3/20*self.field[(x+2) % nx] \
                    + 1/60*self.field[(x+3) % nx]
            ndiff = np.squeeze(ndiff)
            self.assertAlmostEqual(diff_field[x], ndiff)


class DerivativeCheck2d(unittest.TestCase):
    def setUp(self):
        self.nb_pts = [23, 27]
        np.random.seed(7)
        self.field = np.empty(self.nb_pts, order='f')
        self.field[:] = np.random.random(self.nb_pts)
        self.fft = muFFT.FFT(self.nb_pts)
        self.nb_dof = 1
        self.fft.create_plan(self.nb_dof)
        self.fourier_field = self.fft.register_fourier_space_field(
            "fft_workspace", self.nb_dof)
        self.fft.fft(self.field, self.fourier_field)

    def test_rollaxis(self):
        dz = muFFT.DiscreteDerivative([0, 0, 0], [[[-1, 1]]])
        self.assertTrue(dz.stencil.flags.owndata)
        self.assertTrue(np.allclose(dz.stencil[0, 0, :], [-1, 1]))
        self.assertTrue(np.allclose(dz.lbounds, [0, 0, 0]))

        dy = dz.rollaxes(-1)
        self.assertTrue(np.allclose(dy.stencil[0, :, 0], [-1, 1]))
        self.assertTrue(np.allclose(dy.stencil, [[[-1], [1]]]))
        self.assertTrue(np.allclose(dy.lbounds, [0, 0, 0]))

        dx = dy.rollaxes(-1)
        self.assertTrue(np.allclose(dx.stencil[:, 0, 0], [-1, 1]))
        self.assertTrue(np.allclose(dx.stencil, [[[-1]], [[1]]]))
        self.assertTrue(np.allclose(dx.lbounds, [0, 0, 0]))

        dx = muFFT.DiscreteDerivative([0, 0], [[-0.5, 0.5],
                                               [-0.5, 0.5]])
        dy = dx.rollaxes(-1)
        self.assertTrue(np.allclose(dy.stencil.reshape((2, 2)),
                                    [[-0.5, -0.5],
                                     [ 0.5,  0.5]]))

        dz = muFFT.DiscreteDerivative([0, 0, 0],
                                      [[[-0.25, 0.25], [-0.25, 0.25]],
                                       [[-0.25, 0.25], [-0.25, 0.25]]])
        self.assertTrue(np.allclose(dz.stencil[0, 0, :], [-0.25, 0.25]))
        dy = dz.rollaxes(-1)
        self.assertTrue(np.allclose(dy.stencil[0, :, 0], [-0.25, 0.25]))
        dx = dy.rollaxes(-1)
        self.assertTrue(np.allclose(dx.stencil[:, 0, 0], [-0.25, 0.25]))
        self.assertTrue(np.allclose(dy.stencil.reshape((2, 2, 2)),
                                    [[[-0.25, -0.25], [0.25, 0.25]],
                                     [[-0.25, -0.25], [0.25, 0.25]]]))
        self.assertTrue(np.allclose(dx.stencil.reshape((2, 2, 2)),
                                    [[[-0.25, -0.25], [-0.25, -0.25]],
                                     [[ 0.25,  0.25], [ 0.25,  0.25]]]))

    def test_fourier_derivative(self):
        diffop = muFFT.FourierDerivative(2, 0)
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        fourier_field_copy = np.copy(self.fourier_field)
        diff_field = np.zeros_like(self.field, order='f')
        ndiff = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        self.fft.ifft(1j*2*np.pi*q[0] * fourier_field_copy, ndiff)
        ndiff *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        ndiff = np.squeeze(ndiff)
        for x in range(nx):
            for y in range(ny):
                self.assertAlmostEqual(diff_field[x, y], ndiff[x, y])

    def test_fourier_derivative_2_corner(self):
        # shift the fourier derivative into the lower left shift=[-1/6, -1/6]
        # corner. (Here the grid spacing is 1 in each direction, otherwise one
        # should consider it to give the real space shift correct.)
        shift = np.array([-1/6, -1/6])
        diffop = muFFT.FourierDerivative(2, 0, shift)
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        fourier_field_copy = np.copy(self.fourier_field)
        diff_field = np.zeros_like(self.field, order='f')
        ndiff = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        self.fft.ifft(1j*2*np.pi*q[0] *
                      np.exp(1j*2*np.pi*np.einsum("i,i...->...", shift, q)) *
                      fourier_field_copy, ndiff)
        ndiff *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        ndiff = np.squeeze(ndiff)
        for x in range(nx):
            for y in range(ny):
                self.assertAlmostEqual(diff_field[x, y], ndiff[x, y])

    def test_fourier_derivative_2_full(self):
        # shift the fourier derivative by one grid point in x- and y-direction
        # shift=[1, 1]. (Here the grid spacing is 1 in each direction, other-
        # wise one should consider it to give the real space shift correct.)
        shift = np.array([1, 1])
        diffop = muFFT.FourierDerivative(2, 0, shift)
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        fourier_field_copy = np.copy(self.fourier_field)
        diff_field = np.zeros_like(self.field, order='f')
        ndiff = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        self.fft.ifft(1j*2*np.pi*q[0] * fourier_field_copy, ndiff)
        ndiff *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        ndiff = np.squeeze(ndiff)
        for x in range(nx):
            for y in range(ny):
                self.assertAlmostEqual(diff_field[x, y],
                                       ndiff[(x+1) % nx, (y+1) % ny])

    def test_upwind_differences_x(self):
        diffop = muFFT.Stencils2D.upwind_x
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                ndiff = self.field[(x+1) % nx, y] - self.field[x, y]
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_upwind_differences_y(self):
        diffop = muFFT.Stencils2D.upwind_y
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                ndiff = self.field[x, (y+1) % ny] - self.field[x, y]
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_d_11_01(self):
        diffop = muFFT.Stencils2D.d_11_01
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                ndiff = self.field[(x+1) % nx, (y+1) % ny] \
                    - self.field[x, (y+1) % ny]
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_d_11_10(self):
        diffop = muFFT.Stencils2D.d_11_10
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                ndiff = self.field[(x+1) % nx, (y+1) % ny] \
                    - self.field[(x+1) % nx, y]
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_upwind_differences_y2(self):
        diffop = muFFT.DiscreteDerivative([0, 0], [[-1, 1],
                                                   [0,  0]])
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                ndiff = self.field[x, (y+1) % ny] - self.field[x, y]
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_shifted_upwind_differences_y(self):
        diffop = muFFT.DiscreteDerivative([0, 0], [[0,  0],
                                                   [-1, 1]])
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                ndiff = self.field[(x+1) % nx, (y+1) % ny] \
                    - self.field[(x+1) % nx, y]
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_averaged_upwind_differences_x(self):
        diffop = muFFT.Stencils2D.averaged_upwind_x
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                ndiff = (self.field[(x+1) % nx, y] - self.field[x, y]
                         + self.field[(x+1) % nx, (y+1) % ny]
                         - self.field[x, (y+1) % ny])/2
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_averaged_upwind_differences_y(self):
        diffop = muFFT.Stencils2D.averaged_upwind_y
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation

        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                ndiff = (self.field[x, (y+1) % ny] - self.field[x, y]
                         + self.field[(x+1) % nx, (y+1) % ny]
                         - self.field[(x+1) % nx, y])/2
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_central_differences_x(self):
        diffop = muFFT.Stencils2D.central_x
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                ndiff = (self.field[(x+1) % nx, y]
                         - self.field[(x-1) % nx, y])/2
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_central_differences_y(self):
        diffop = muFFT.Stencils2D.central_y
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                ndiff = (self.field[x, (y+1) % ny]
                         - self.field[x, (y-1) % ny])/2
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_sixth_order_central_differences(self):
        diffop = muFFT.DiscreteDerivative([0, -3],
                                          [[-1/60, 3/20, -3/4, 0,
                                            3/4, -3/20, 1/60]])
        self.assertTrue(np.allclose(diffop.lbounds, [0, -3]))
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                ndiff = -1/60*self.field[x, (y-3) % ny] \
                    + 3/20*self.field[x, (y-2) % ny] \
                    - 3/4*self.field[x, (y-1) % ny] \
                    + 3/4*self.field[x, (y+1) % ny] \
                    - 3/20*self.field[x, (y+2) % ny] \
                    + 1/60*self.field[x, (y+3) % ny]
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_central_2nd_differences_x(self):
        diffop = muFFT.Stencils2D.central_2nd_x
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny = self.nb_pts

    def test_convenience_interface_discrete_deriv(self):
        assert self.nb_dof == 1
        diffop = muFFT.Stencils2D.upwind_x
        field = self.field.reshape(self.nb_pts, order='F')
        diff_field = diffop.apply(field)
        print('diff_field[0, 0] (python):', diff_field[0, 0])
        diff_field = np.squeeze(diff_field)
        nx, ny = self.nb_pts
        for x in range(nx):
            for y in range(ny):
                ndiff = self.field[(x+1) % nx, y] - self.field[x, y]
                if x == 0 and y == 0:
                    print('ndiff:', ndiff)
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_central_2nd_differences_y(self):
        diffop = muFFT.Stencils2D.central_2nd_y
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                ndiff = self.field[x, (y+1) % ny] - 2*self.field[x, y] \
                    + self.field[x, (y-1) % ny]
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_hexagonal_T1_dx(self):
        diffop = muFFT.Stencils2D.hexagonal_T1_dx
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                ndiff = self.field[(x+1) % nx, y] - self.field[x, y]
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_hexagonal_T1_dy(self):
        diffop = muFFT.Stencils2D.hexagonal_T1_dy
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                ndiff = 1/2*(2*self.field[x, (y+1) % ny] - self.field[x, y]
                             - self.field[(x+1) % nx, y])
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_hexagonal_T2_dx(self):
        diffop = muFFT.Stencils2D.hexagonal_T2_dx
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                ndiff = self.field[(x+1) % nx, (y+1) % ny] \
                    - self.field[x, (y+1) % ny]
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_hexagonal_T2_dy(self):
        diffop = muFFT.Stencils2D.hexagonal_T2_dy
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                ndiff = 1/2*(-2*self.field[(x+1) % nx, y]
                             + self.field[x, (y+1) % ny]
                             + self.field[(x+1) % nx, (y+1) % ny])
                ndiff = np.squeeze(ndiff)
                self.assertAlmostEqual(diff_field[x, y], ndiff)


class DerivativeCheck3d(unittest.TestCase):
    def setUp(self):
        self.nb_pts = [23, 23, 17]
        self.field = np.random.random(self.nb_pts)
        self.fft = muFFT.FFT(self.nb_pts)
        self.nb_dof = 1
        self.fft.create_plan(self.nb_dof)
        self.fourier_field = self.fft.register_fourier_space_field(
            "fft_workspace", self.nb_dof)
        self.fft.fft(self.field, self.fourier_field)

    def test_upwind_differences_x(self):
        diffop = muFFT.Stencils3D.upwind_x
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[(x+1) % nx, y, z] - self.field[x, y, z]
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_upwind_differences_y(self):
        diffop = muFFT.Stencils3D.upwind_y
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[x, (y+1) % ny, z] - self.field[x, y, z]
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_upwind_differences_z(self):
        diffop = muFFT.Stencils3D.upwind_z
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[x, y, (z+1) % nz] - self.field[x, y, z]
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_d_100_000(self):
        diffop = muFFT.Stencils3D.d_100_000
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[(x+1) % nx, y, z] - self.field[x, y, z]
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_d_110_010(self):
        diffop = muFFT.Stencils3D.d_110_010
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[(x+1) % nx, (y+1) % ny, z] \
                        - self.field[x, (y+1) % ny, z]
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_d_111_011(self):
        diffop = muFFT.Stencils3D.d_111_011
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[(x+1) % nx, (y+1) % ny, (z+1) % nz] \
                        - self.field[x, (y+1) % ny, (z+1) % nz]
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_d_101_001(self):
        diffop = muFFT.Stencils3D.d_101_001
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[(x+1) % nx, y, (z+1) % nz] \
                        - self.field[x, y, (z+1) % nz]
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_d_010_000(self):
        diffop = muFFT.Stencils3D.d_010_000
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[x, (y+1) % ny, z] - self.field[x, y, z]
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_d_110_100(self):
        diffop = muFFT.Stencils3D.d_110_100
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[(x+1) % nx, (y+1) % ny, z] \
                        - self.field[(x+1) % nx, y, z]
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_d_111_101(self):
        diffop = muFFT.Stencils3D.d_111_101
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[(x+1) % nx, (y+1) % ny, (z+1) % nz] \
                        - self.field[(x+1) % nx, y, (z+1) % nz]
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_d_011_001(self):
        diffop = muFFT.Stencils3D.d_011_001
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[x, (y+1) % ny, (z+1) % nz] \
                        - self.field[x, y, (z+1) % nz]
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_d_001_000(self):
        diffop = muFFT.Stencils3D.d_001_000
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[x, y, (z+1) % nz] - self.field[x, y, z]
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_d_101_100(self):
        diffop = muFFT.Stencils3D.d_101_100
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[(x+1) % nx, y, (z+1) % nz] \
                        - self.field[(x+1) % nx, y, z]
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_d_111_110(self):
        diffop = muFFT.Stencils3D.d_111_110
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[(x+1) % nx, (y+1) % ny, (z+1) % nz] \
                        - self.field[(x+1) % nx, (y+1) % ny, z]
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_d_011_010(self):
        diffop = muFFT.Stencils3D.d_011_010
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[x, (y+1) % ny, (z+1) % nz] \
                        - self.field[x, (y+1) % ny, z]
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_averaged_upwind_differences_x(self):
        diffop = muFFT.Stencils3D.averaged_upwind_x
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = (self.field[(x+1) % nx, y, z] - self.field[x, y, z]
                             + self.field[(x+1) % nx, (y+1) % ny, z]
                             - self.field[x, (y+1) % ny, z]
                             + self.field[(x+1) % nx, y, (z+1) % nz]
                             - self.field[x, y, (z+1) % nz]
                             + self.field[(x+1) % nx, (y+1) % ny, (z+1) % nz]
                             - self.field[x, (y+1) % ny, (z+1) % nz])/4
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_averaged_upwind_differences_y(self):
        diffop = muFFT.Stencils3D.averaged_upwind_y
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = (self.field[x, (y+1) % ny, z]
                             - self.field[x, y, z]
                             + self.field[(x+1) % nx, (y+1) % ny, z]
                             - self.field[(x+1) % nx, y, z]
                             + self.field[x, (y+1) % ny, (z+1) % nz]
                             - self.field[x, y, (z+1) % nz]
                             + self.field[(x+1) % nx, (y+1) % ny, (z+1) % nz]
                             - self.field[(x+1) % nx, y, (z+1) % nz])/4
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_averaged_upwind_differences_z(self):
        diffop = muFFT.Stencils3D.averaged_upwind_z
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = (self.field[x, y, (z+1) % nz]
                             - self.field[x, y, z]
                             + self.field[(x+1) % nx, y, (z+1) % nz]
                             - self.field[(x+1) % nx, y, z]
                             + self.field[x, (y+1) % ny, (z+1) % nz]
                             - self.field[x, (y+1) % ny, z]
                             + self.field[(x+1) % nx, (y+1) % ny, (z+1) % nz]
                             - self.field[(x+1) % nx, (y+1) % ny, z])/4
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_central_differences_x(self):
        diffop = muFFT.Stencils3D.central_x
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = (self.field[(x+1) % nx, y, z]
                             - self.field[(x-1) % nx, y, z])/2
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_central_differences_y(self):
        diffop = muFFT.Stencils3D.central_y
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = (self.field[x, (y+1) % ny, z]
                             - self.field[x, (y-1) % ny, z])/2
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_central_differences_z(self):
        diffop = muFFT.Stencils3D.central_z
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = (self.field[x, y, (z+1) % nz]
                             - self.field[x, y, (z-1) % nz])/2
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_convenience_interface_discrete_deriv(self):
        assert self.nb_dof == 1
        diffop = muFFT.Stencils3D.central_z
        field = self.field.reshape([*self.nb_pts], order='F')
        diff_field = diffop.apply(field)
        diff_field = np.squeeze(diff_field)
        nx, ny, nz = self.nb_pts
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = (self.field[x, y, (z+1) % nz]
                             - self.field[x, y, (z-1) % nz])/2
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_five_tetraheda_T0_derivative_x(self):
        diffop = muFFT.Stencils3D.linear_finite_elements_5[0]
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = (self.field[(x+1) % nx, (y+1) % ny, (z+1) % nz]
                             + self.field[(x+1) % nx, y, z]
                             - self.field[x, (y+1) % ny, z]
                             - self.field[x, y, (z+1) % nz])/2
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_five_tetraheda_T0_derivative_y(self):
        diffop = muFFT.Stencils3D.linear_finite_elements_5[1]
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = (self.field[(x+1) % nx, (y+1) % ny, (z+1) % nz]
                             - self.field[(x+1) % nx, y, z]
                             + self.field[x, (y+1) % ny, z]
                             - self.field[x, y, (z+1) % nz])/2
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_five_tetraheda_T0_derivative_z(self):
        diffop = muFFT.Stencils3D.linear_finite_elements_5[2]
        q = self.fft.fftfreq
        d = diffop.fourier(q)
        diff_field = np.zeros_like(self.field, order='f')
        self.fft.ifft(d * self.fourier_field, diff_field)
        diff_field *= self.fft.normalisation
        nx, ny, nz = self.nb_pts
        diff_field = np.squeeze(diff_field)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = (self.field[(x+1) % nx, (y+1) % ny, (z+1) % nz]
                             - self.field[(x+1) % nx, y, z]
                             - self.field[x, (y+1) % ny, z]
                             + self.field[x, y, (z+1) % nz])/2
                    ndiff = np.squeeze(ndiff)
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_six_regular_tetraheda(self):
        def get_diff_field(element, direction):
            deriv_n = 3*element + direction
            diffop = muFFT.Stencils3D.linear_finite_elements_6_regular[deriv_n]
            q = self.fft.fftfreq
            d = diffop.fourier(q)
            diff_field = np.zeros_like(self.field, order='f')
            self.fft.ifft(d * self.fourier_field, diff_field)
            diff_field *= self.fft.normalisation
            diff_field = np.squeeze(diff_field)

            return diff_field

        direction_str = {"0": "x",
                         "1": "y",
                         "2": "z"}

        def error_message(element, direction):
            print(f"Derivative test for tetrahedra {element} fails in "
                  f"{direction_str[str(direction)]}-direction.")

        def test_derivative(diff_field, element, direction):
            s = stencils[str(element)+str(direction)]

            # derivative in x-direction
            if direction == 0:
                for x, y, z in itertools.product(
                        range(nx), range(ny), range(nz)):
                    ndiff = self.field[(x+s[0, 0]) % nx,
                                       (y+s[0, 1]) % ny,
                                       (z+s[0, 2]) % nz]
                    ndiff -= self.field[(x+s[1, 0]) % nx,
                                        (y+s[1, 1]) % ny,
                                        (z+s[1, 2]) % nz]
                    ndiff += 1/2*self.field[(x+s[2, 0]) % nx,
                                            (y+s[2, 1]) % ny,
                                            (z+s[2, 2]) % nz]
                    ndiff -= 1/2*self.field[(x+s[3, 0]) % nx,
                                            (y+s[3, 1]) % ny,
                                            (z+s[3, 2]) % nz]

                    ndiff = np.squeeze(ndiff)
                    if round(diff_field[x, y, z], 7) - round(ndiff, 7) != 0:
                        error_message(element, direction)
                        self.assertAlmostEqual(diff_field[x, y, z], ndiff)
                        continue  # skipp test for other field values
            # derivative in y-direction
            elif direction == 1:
                for x, y, z in itertools.product(
                        range(nx), range(ny), range(nz)):
                    ndiff = self.field[(x+s[0, 0]) % nx,
                                       (y+s[0, 1]) % ny,
                                       (z+s[0, 2]) % nz]
                    ndiff -= self.field[(x+s[1, 0]) % nx,
                                        (y+s[1, 1]) % ny,
                                        (z+s[1, 2]) % nz]
                    ndiff += 2/3*self.field[(x+s[2, 0]) % nx,
                                            (y+s[2, 1]) % ny,
                                            (z+s[2, 2]) % nz]
                    ndiff -= 2/3*self.field[(x+s[3, 0]) % nx,
                                            (y+s[3, 1]) % ny,
                                            (z+s[3, 2]) % nz]

                    ndiff = np.squeeze(ndiff)
                    if round(diff_field[x, y, z], 7) - round(ndiff, 7) != 0:
                        error_message(element, direction)
                        self.assertAlmostEqual(diff_field[x, y, z], ndiff)
                        continue  # skipp test for other field values
            # derivative in z-direction
            elif direction == 2:
                for x, y, z in itertools.product(
                        range(nx), range(ny), range(nz)):
                    ndiff = self.field[(x+s[0, 0]) % nx,
                                       (y+s[0, 1]) % ny,
                                       (z+s[0, 2]) % nz]
                    ndiff -= self.field[(x+s[1, 0]) % nx,
                                        (y+s[1, 1]) % ny,
                                        (z+s[1, 2]) % nz]

                    ndiff = np.squeeze(ndiff)
                    if round(diff_field[x, y, z], 7) - round(ndiff, 7) != 0:
                        error_message(element, direction)
                        self.assertAlmostEqual(diff_field[x, y, z], ndiff)
                        continue  # skipp test for other field values

        nx, ny, nz = self.nb_pts
        stencils = {
            "00": np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]]),
            "01": np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]),
            "02": np.array([[1, 1, 1], [1, 1, 0]]),
            "10": np.array([[1, 0, 0], [0, 0, 0], [1, 0, 1], [1, 1, 1]]),
            "11": np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 1]]),
            "12": np.array([[1, 0, 1], [1, 0, 0]]),
            "20": np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0]]),
            "21": np.array([[0, 1, 0], [0, 0, 0], [1, 1, 0], [1, 1, 1]]),
            "22": np.array([[1, 1, 1], [1, 1, 0]]),
            "30": np.array([[1, 1, 1], [0, 1, 1], [0, 0, 0], [0, 1, 0]]),
            "31": np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 1, 1]]),
            "32": np.array([[0, 1, 1], [0, 1, 0]]),
            "40": np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]),
            "41": np.array([[1, 1, 1], [1, 0, 1], [0, 0, 0], [0, 0, 1]]),
            "42": np.array([[0, 0, 1], [0, 0, 0]]),
            "50": np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 1, 1]]),
            "51": np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0], [0, 0, 1]]),
            "52": np.array([[0, 0, 1], [0, 0, 0]])
        }

        for element, direction in itertools.product(range(6), range(3)):
            diff_field = get_diff_field(element, direction)
            test_derivative(diff_field, element, direction)


if __name__ == "__main__":
    unittest.main()
