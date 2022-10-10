#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_projection_tests.py

@author Till Junge <till.junge@altermail.ch>

@date   18 Jan 2018

@brief  compare µSpectre's projection operators to GooseFFT

Copyright © 2018 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µSpectre; see the file COPYING. If not, write to the
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

from python_test_imports import µ, muFFT, muGrid
from python_goose_ref import SmallStrainProjectionGooseFFT,\
    FiniteStrainProjectionGooseFFT
import _muSpectre
import muSpectre as msp

from muFFT import Stencils2D


def build_single_quad_pt_check(Projection, RefProjection, name,
                               formulation, do_factory_test=False):
    class ProjectionSingleQuadPtCheck(unittest.TestCase):
        def __init__(self, methodName='runTest'):
            super().__init__(methodName)
            self.__class__.__qualname__ = name

        def setUp(self):
            self.ref = RefProjection
            self.nb_grid_pts = self.ref.nb_grid_pts
            self.ndim = self.ref.ndim
            self.shape = list((self.nb_grid_pts for _ in range(self.ndim)))
            self.fft = muFFT.FFT(self.shape)
            self.fft.create_plan(self.ndim * self.ndim)
            self.projection = Projection(
                self.fft, [float(x) for x in self.shape])
            self.projection.initialise()
            self.tol = 1e-12*np.prod(self.shape)

        def test_factory(self):
            if do_factory_test:
                projection = µ.Projection(self.shape,
                                          [float(x) for x in self.shape],
                                          formulation=formulation)
                self.assertTrue(
                    projection.__class__ == self.projection.__class__)

        def test_CompareGhat4(self):
            # refG is rowmajor and the dims are i,j,k,l,x,y(,z)
            # reshape refG so they are n² × n² × ¶(nb_grid_pts)
            order = list(range(self.ndim+4))
            order[-self.ndim:] = reversed(order[-self.ndim:])
            refG = self.ref.Ghat4.transpose(*order).reshape(
                self.ndim**2, self.ndim**2, np.prod(self.shape))
            # mspG is colmajor (not sure what that's worth, though) with dims
            # ijkl, xy(z)
            # reshape mspG so they are ¶(hermitian) × n² × n²
            ref_sizes = self.shape
            msp_sizes = muFFT.get_nb_hermitian_grid_pts(self.shape)
            hermitian_size = np.prod(msp_sizes)
            mspG = self.projection.operator
            # this test only makes sense for fully stored ghats (i.e.,
            # not for the faster alternative implementation
            if mspG.size != hermitian_size*self.ndim**4:
                return

            for i in range(hermitian_size):
                coord = µ.get_domain_ccoord(msp_sizes, i)
                ref_id = µ.get_domain_index(ref_sizes, coord)
                msp_id = µ.get_domain_index(msp_sizes, coord)
                # story behind this order vector:
                # There was this issue with the projection operator of
                # de Geus acting on the the transpose of the gradient.
                order = np.arange(self.ndim**2).reshape(
                    self.ndim, self.ndim).T.reshape(-1)
                msp_g = mspG[:, msp_id].reshape(
                    self.ndim**2, self.ndim**2)[order, :]
                error = np.linalg.norm(refG[:, :, ref_id] -
                                       msp_g)
                condition = error < self.tol
                if not condition:
                    print("G_µ{}, at index {} =\n{}".format(
                        coord, msp_id, msp_g))
                    print("G_g{}, at index {} =\n{}".format(
                        coord, ref_id, refG[:, :, ref_id]))
                self.assertTrue(condition)

        def test_projection_result(self):
            # create a bogus strain field in GooseFFT format
            # dim × dim × N × N (× N)
            strain_shape = (self.ndim, self.ndim, *self.shape)
            strain = np.arange(np.prod(strain_shape)).reshape(strain_shape)
            # if we're testing small strain projections, it needs to be symmetric
            if self.projection.formulation == µ.Formulation.small_strain:
                strain += strain.transpose(1, 0, *range(2, len(strain.shape)))
            b_g = self.ref.G(strain).reshape(strain.shape)
            b_µ = self.projection.apply_projection(strain)

            assert np.allclose(b_g, b_µ)

        def test_integration_result(self):
            # random slope
            mean_strain = np.random.random((self.ndim, self.ndim))
            # create a random displacement field
            disp = np.random.random((self.ndim, *self.shape))
            # make sure mean is zero for each Cartesian direction
            for i in range(self.ndim):
                disp[i] -= disp[i].mean()
            # compute derivative
            q = self.fft.fftfreq
            strain_before_projection = \
                np.swapaxes([self.fft.normalisation *
                             self.fft.ifft(g.fourier(q) * self.fft.fft(disp))
                             for g in self.projection.gradient],
                            0, 1)
            # symmetrize for small strain
            if self.projection.formulation == µ.Formulation.small_strain:
                strain_before_projection = 0.5*(
                        strain_before_projection +
                        np.swapaxes(strain_before_projection, 0, 1))
            strain = self.projection.apply_projection(strain_before_projection)
            self.assertTrue(np.allclose(strain, strain_before_projection))
            strain = (strain.T + mean_strain.T).T
            recon_disp = self.projection.integrate(strain).squeeze()

            coords = np.array(np.meshgrid(*(np.arange(x) for x in self.shape), indexing='ij'))
            disp += coords.T.dot(mean_strain.T).T

            self.assertTrue(np.allclose(recon_disp, disp))

    return ProjectionSingleQuadPtCheck


def build_multiple_quad_pt_check(Projection1q, Projection2q, name):
    class ProjectionMultipleQuadPtsCheck(unittest.TestCase):
        def __init__(self, methodName='runTest'):
            super().__init__(methodName)
            self.__class__.__qualname__ = name

        def setUp(self):
            self.nb_grid_pts = [3, 3]

        def test_single_quad_pts(self):
            def compute_gradient():
                for dim in range(nb_dim):
                    for d in range(nb_dim):
                        gradient[d].apply(in_field, dim, out_field,
                                            dim + nb_dim * d)

            nb_dim = len(self.nb_grid_pts)
            gradient = [Stencils2D.upwind_x, Stencils2D.upwind_y]

            fft = muFFT.FFT(self.nb_grid_pts)
            projection = Projection1q(fft, [float(x) for x in self.nb_grid_pts],
                                      gradient, [1])
            projection.initialise()

            coll = muGrid.GlobalFieldCollection(nb_dim)
            coll.initialise(self.nb_grid_pts, self.nb_grid_pts)

            in_field = coll.register_real_field("in", 2)
            in_arr = in_field.array(muGrid.Pixel)
            in_arr[0, 0, 0] = 1.0

            out_field = coll.register_real_field("out", 4)
            out_arr = out_field.array(muGrid.Pixel)
            compute_gradient()

            self.assertTrue(np.allclose(out_arr[0], [[-1, 0, 0],
                                                      [0, 0, 0],
                                                      [1, 0, 0]]))
            self.assertTrue(np.allclose(out_arr[1], np.zeros_like(out_arr[1])))
            self.assertTrue(np.allclose(out_arr[2], [[-1, 0, 1],
                                                      [0, 0, 0],
                                                      [0, 0, 0]]))
            self.assertTrue(np.allclose(out_arr[3], np.zeros_like(out_arr[3])))

            if projection.formulation == µ.Formulation.small_strain:
                # symmetrize strain tensor
                out_arr[1] = out_arr[2] = 0.5*(out_arr[1] + out_arr[2])

            ref_field = coll.register_real_field("ref", 4)
            ref_arr = ref_field.array(muGrid.Pixel)
            ref_arr[...] = out_arr[...]

            projection.apply_projection(out_field)

            self.assertTrue(np.allclose(out_arr, ref_arr))

        def test_double_quad_pts(self):
            def compute_gradient():
                for dim in range(nb_dim):
                    for d in range(2*nb_dim):
                        gradient[d].apply(in_field, dim, out_field,
                                            dim + nb_dim * d)

            nb_dim = len(self.nb_grid_pts)
            gradient = [Stencils2D.d_10_00, Stencils2D.d_01_00,
                        Stencils2D.d_11_01, Stencils2D.d_11_10]

            fft = muFFT.FFT(self.nb_grid_pts)
            projection = Projection2q(
                fft, [float(x) for x in self.nb_grid_pts],
                gradient, [1, 1])
            projection.initialise()

            coll = muGrid.GlobalFieldCollection(nb_dim)
            coll.initialise(self.nb_grid_pts, self.nb_grid_pts)

            in_field = coll.register_real_field("in", 2)
            in_arr = in_field.array(muGrid.Pixel)
            in_arr[0, 0, 0] = 1.0

            out_field = coll.register_real_field("out", 8)
            out_arr = out_field.array(muGrid.Pixel)
            compute_gradient()

            self.assertTrue(np.allclose(out_arr[0], [[-1, 0, 0],
                                                     [0, 0, 0],
                                                     [1, 0, 0]]))
            self.assertTrue(np.allclose(out_arr[1], np.zeros_like(out_arr[1])))
            self.assertTrue(np.allclose(out_arr[2], [[-1, 0, 1],
                                                     [0, 0, 0],
                                                     [0, 0, 0]]))
            self.assertTrue(np.allclose(out_arr[3], np.zeros_like(out_arr[3])))
            self.assertTrue(np.allclose(out_arr[4], [[0, 0, -1],
                                                     [0, 0, 0],
                                                     [0, 0, 1]]))
            self.assertTrue(np.allclose(out_arr[5], np.zeros_like(out_arr[5])))
            self.assertTrue(np.allclose(out_arr[6], [[0, 0, 0],
                                                     [0, 0, 0],
                                                     [-1, 0, 1]]))
            self.assertTrue(np.allclose(out_arr[7], np.zeros_like(out_arr[7])))

            if projection.formulation == µ.Formulation.small_strain:
                # symmetrize strain tensor
                out_arr[1] = out_arr[2] = 0.5*(out_arr[1] + out_arr[2])
                out_arr[5] = out_arr[6] = 0.5*(out_arr[5] + out_arr[6])

            ref_field = coll.register_real_field("ref", 8)
            ref_arr = ref_field.array(muGrid.Pixel)
            ref_arr[...] = out_arr[...]

            projection.apply_projection(out_field)

            self.assertTrue(np.allclose(out_arr, ref_arr))

        def test_double_quad_pts_convenience_interface(self):
            def compute_gradient():
                for dim in range(nb_dim):
                    for d in range(2*nb_dim):
                        gradient[d].apply(in_field, dim, out_field,
                                          dim + nb_dim * d)

            nb_dim = len(self.nb_grid_pts)
            gradient = [Stencils2D.d_10_00, Stencils2D.d_01_00,
                        Stencils2D.d_11_01, Stencils2D.d_11_10]

            coll = muGrid.GlobalFieldCollection(nb_dim)
            coll.initialise(self.nb_grid_pts, self.nb_grid_pts)

            in_field = coll.register_real_field("in", 2)
            in_arr = in_field.array(muGrid.Pixel)
            in_arr[0, 0, 0] = 1.0

            out_field = coll.register_real_field("out", 8)
            out_arr = out_field.array(muGrid.Pixel)
            compute_gradient()

            self.assertTrue(np.allclose(out_arr[0], [[-1, 0, 0],
                                                     [0, 0, 0],
                                                     [1, 0, 0]]))
            self.assertTrue(np.allclose(out_arr[1], np.zeros_like(out_arr[1])))
            self.assertTrue(np.allclose(out_arr[2], [[-1, 0, 1],
                                                     [0, 0, 0],
                                                     [0, 0, 0]]))
            self.assertTrue(np.allclose(out_arr[3], np.zeros_like(out_arr[3])))
            self.assertTrue(np.allclose(out_arr[4], [[0, 0, -1],
                                                     [0, 0, 0],
                                                     [0, 0, 1]]))
            self.assertTrue(np.allclose(out_arr[5], np.zeros_like(out_arr[5])))
            self.assertTrue(np.allclose(out_arr[6], [[0, 0, 0],
                                                     [0, 0, 0],
                                                     [-1, 0, 1]]))
            self.assertTrue(np.allclose(out_arr[7], np.zeros_like(out_arr[7])))

            out_arr.shape = (2, 2, 2) + tuple(self.nb_grid_pts)
            grad_arr = out_arr.copy()

            projection = µ.Projection(self.nb_grid_pts,
                                      [float(x) for x in self.nb_grid_pts],
                                      gradient=gradient, weights=[1, 1])
            projection.initialise()
            projection.apply_projection(grad_arr)

            self.assertTrue(np.allclose(out_arr, grad_arr))

    return ProjectionMultipleQuadPtsCheck


def build_5_tet_gauss_weight_check(formulation, name):
    class Projection5TetWeightCheck(unittest.TestCase):
        def __init__(self, methodName='runTest'):
            super().__init__(methodName)
            self.__class__.__qualname__ = name

        def setUp(self):
            self.nb_grid_pts = [2, 2, 2]
            self.lengths = self.nb_grid_pts
            self.dim = len(self.nb_grid_pts)
            self.Youngs_modulus = 1
            self.Poisson_ratio = 0.33
            self.formulation = formulation
            self.gradient, self.weights = \
                msp.linear_finite_elements.gradient_3d_5tet
            self.nb_qpts = len(self.weights)

            # solver
            self.cg_tol = 1e-6
            self.newton_tol = 1e-6
            self.equilibrium_tol = self.newton_tol
            self.verbosity = msp.Verbosity.Silent
            self.maxiter = 100

        def test_for_equal_stress_and_strain(self):
            """
            check if the stress and strain in each element of a voxel is equal.
            """
            def setup_and_run_simulation(weights):
                cell = msp.Cell(self.nb_grid_pts, self.lengths,
                                self.formulation, self.gradient, weights)
                mat = msp.material.MaterialLinearElastic1_3d.make(
                    cell, "single_linear_elastic_voxel",
                    self.Youngs_modulus, self.Poisson_ratio)
                for pixel_id, _ in cell.pixels.enumerate():
                    mat.add_pixel(pixel_id)

                solver = msp.solvers.KrylovSolverCG(cell, self.cg_tol,
                                                    self.maxiter,
                                                    self.verbosity)
                DelF = np.array([[0, 0.1, 0],
                                 [0, 0.2, 0.02],
                                 [0.3, 0, 0.01]])
                if self.formulation == msp.Formulation.small_strain:
                    DelF = 1/2*(DelF + DelF.T)
                res = msp.solvers.newton_cg(cell, DelF, solver,
                                            self.newton_tol,
                                            self.equilibrium_tol,
                                            self.verbosity)
                strain = res.grad.reshape((self.dim, self.dim,
                                           self.nb_qpts, -1),
                                          order="F").T.swapaxes(2, 3)
                stress = res.stress.reshape((self.dim, self.dim,
                                             self.nb_qpts, -1),
                                            order="F").T.swapaxes(2, 3)

                return stress, strain

            # give correct weights
            stress, strain = setup_and_run_simulation(weights=self.weights)
            self.assertTrue(np.allclose(strain, np.roll(strain, 1, axis=1)))
            self.assertTrue(np.allclose(stress, np.roll(stress, 1, axis=1)))

            # give wrong weights
            stress, strain = setup_and_run_simulation(weights=[1]*self.nb_qpts)

            # Also with wrong weights the result is correct! This is the case
            # because of the strong confinement for a single pixel
            self.assertTrue(np.allclose(strain, np.roll(strain, 1, axis=1)))
            self.assertTrue(np.allclose(stress, np.roll(stress, 1, axis=1)))

    return Projection5TetWeightCheck


def get_goose(ndim, proj_type): return proj_type(
    ndim, 5, 2, 70e9, .33, 3.)


def get_finite_goose(ndim): return get_goose(
    ndim, FiniteStrainProjectionGooseFFT)


def get_small_goose(ndim): return get_goose(
    ndim,  SmallStrainProjectionGooseFFT)


# Single quadrature point checks
small_default_3 = build_single_quad_pt_check(
    _muSpectre.ProjectionSmallStrain_3d,
    get_small_goose(3),
    "SmallStrainDefaultProjection3d",
    µ.Formulation.small_strain,
    True)
small_default_2 = build_single_quad_pt_check(
    _muSpectre.ProjectionSmallStrain_2d,
    get_small_goose(2),
    "SmallStrainDefaultProjection2d",
    µ.Formulation.small_strain,
    True)

finite_default_3 = build_single_quad_pt_check(
    _muSpectre.ProjectionFiniteStrain_3d,
    get_finite_goose(3),
    "FiniteStrainDefaultProjection3d",
    µ.Formulation.finite_strain,
    False)
finite_default_2 = build_single_quad_pt_check(
    _muSpectre.ProjectionFiniteStrain_2d,
    get_finite_goose(2),
    "FiniteStrainDefaultProjection2d",
    µ.Formulation.finite_strain,
    False)

finite_fast_3 = build_single_quad_pt_check(
    _muSpectre.ProjectionFiniteStrainFast_3d,
    get_finite_goose(3),
    "FiniteStrainFastProjection3d",
    µ.Formulation.finite_strain,
    True)
finite_fast_2 = build_single_quad_pt_check(
    _muSpectre.ProjectionFiniteStrainFast_2d,
    get_finite_goose(2),
    "FiniteStrainFastProjection2d",
    µ.Formulation.finite_strain,
    True)

# Multiple quadrature points check
multi_small_default_2 = build_multiple_quad_pt_check(
    _muSpectre.ProjectionSmallStrain_2d,
    _muSpectre.ProjectionSmallStrain_2q_2d,
    "SmallStrainDefaultProjection2d")

multi_finite_default_2 = build_multiple_quad_pt_check(
    _muSpectre.ProjectionFiniteStrain_2d,
    _muSpectre.ProjectionFiniteStrain_2q_2d,
    "FiniteStrainDefaultProjection2d")

multi_finite_fast_2 = build_multiple_quad_pt_check(
    _muSpectre.ProjectionFiniteStrainFast_2d,
    _muSpectre.ProjectionFiniteStrainFast_2q_2d,
    "FiniteStrainFastProjection2d")

# five tetrahedra gauss weights check
five_tet_fs = build_5_tet_gauss_weight_check(
    formulation=msp.Formulation.finite_strain,
    name="FiniteStrain5Tet")

five_tet_ss = build_5_tet_gauss_weight_check(
    formulation=msp.Formulation.small_strain,
    name="SmallStrain5Tet")

if __name__ == "__main__":
    unittest.main()
