#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_material_linear_elastic4_test.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   27 Mar 2018

@brief  description

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

from python_test_imports import µ


class MaterialLinearElastic4_Check(unittest.TestCase):
    """
    Check the implementation of storing the first and second Lame constant in
    each cell. Assign the same Youngs modulus and Poisson ratio to each cell,
    from which the two Lame constants are internally computed. Then calculate
    the stress and compare the result with stress=2*mu*Del0 (Hooke law for small
    symmetric strains).
    """

    def setUp(self):
        self.nb_grid_pts = [7, 7]
        self.lengths = [2.3, 3.9]
        self.formulation = µ.Formulation.small_strain
        self.dim = len(self.lengths)

    def test_solver(self):
        Youngs_modulus = 10.
        Poisson_ratio = 0.3

        cell = µ.Cell(self.nb_grid_pts, self.lengths, self.formulation)
        mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")

        for i in cell.pixel_indices:
            mat.add_pixel(i, Youngs_modulus, Poisson_ratio)

        cell.initialise()
        tol = 1e-6
        Del0 = np.array([[0, 0.025],
                         [0.025,  0]])
        maxiter = 100
        verbose = µ.Verbosity.Silent

        solver = µ.solvers.KrylovSolverCG(
            cell, tol, maxiter, verbose)
        r = µ.solvers.newton_cg(
            cell, Del0, solver, tol, tol, verbose)

        # compare the computed stress with the trivial by hand computed stress
        mu = (Youngs_modulus/(2*(1+Poisson_ratio)))
        stress = 2*mu*Del0

        self.assertLess(np.linalg.norm(r.stress.reshape(-1, self.dim**2) -
                                       stress.reshape(1,self.dim**2)), 1e-8)

    def test_tangent(self):
        Youngs_modulus = 10.*(1 + 0.1*np.random.random(np.prod(self.nb_grid_pts)))
        Poisson_ratio  = 0.3*(1 + 0.1*np.random.random(np.prod(self.nb_grid_pts)))

        cell = µ.Cell(self.nb_grid_pts, self.lengths, self.formulation)
        mat = µ.material.MaterialLinearElastic4_2d.make(cell,
                                                        "material")

        for i in cell.pixel_indices:
            mat.add_pixel(i, Youngs_modulus[i], Poisson_ratio[i])

        cell.initialise()
        tol = 1e-6
        Del0 = np.array([[0.1, 0.05],
                         [0.05,  -0.02]])
        maxiter = 100
        verbose = µ.Verbosity.Silent

        solver = µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
        r = µ.solvers.newton_cg(cell, Del0,
                                solver, tol, tol, verbose)

        ### Compute tangent through a finite differences approximation

        F = cell.strain.array()
        stress, tangent = cell.evaluate_stress_tangent(F)

        numerical_tangent = np.zeros_like(tangent)

        eps = 1e-4
        for i in range(2):
            for j in range(2):
                F[i, j] += eps
                stress_plus = cell.evaluate_stress(F).copy()
                F[i, j] -= 2*eps
                stress_minus = cell.evaluate_stress(F).copy()
                F[i, j] += eps
                numerical_tangent[i, j] = (stress_plus - stress_minus)/(2*eps)

        self.assertTrue(np.allclose(tangent, numerical_tangent))

    def test_setter_and_getter_functions(self):
        nb_pts = np.prod(self.nb_grid_pts)
        Youngs_modulus = np.arange(nb_pts)
        Poisson_ratio = np.arange(nb_pts) / (2 * nb_pts)

        cell = µ.Cell(self.nb_grid_pts, self.lengths, self.formulation)
        mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")

        for i in cell.pixel_indices:
            mat.add_pixel(i, Youngs_modulus[i], Poisson_ratio[i])
        cell.initialise()

        random_quad_pt_id_1 = np.random.randint(0, nb_pts)
        self.assertAlmostEqual(random_quad_pt_id_1/(2*nb_pts),
                               mat.get_poisson_ratio(random_quad_pt_id_1),
                               places=8)
        self.assertAlmostEqual(random_quad_pt_id_1,
                               mat.get_youngs_modulus(random_quad_pt_id_1),
                               places=8)

        random_quad_pt_id_2 = np.random.randint(0, nb_pts)
        new_youngs_modulus = 2 * nb_pts
        mat.set_youngs_modulus(random_quad_pt_id_2, new_youngs_modulus)
        self.assertAlmostEqual(random_quad_pt_id_2/(2*nb_pts),
                               mat.get_poisson_ratio(random_quad_pt_id_2),
                               places=8)
        self.assertAlmostEqual(new_youngs_modulus,
                               mat.get_youngs_modulus(random_quad_pt_id_2),
                               places=8)

        random_quad_pt_id_3 = np.random.randint(0, nb_pts)
        new_poisson_ratio = -0.5
        mat.set_poisson_ratio(random_quad_pt_id_3, new_poisson_ratio)
        self.assertAlmostEqual(new_poisson_ratio,
                               mat.get_poisson_ratio(random_quad_pt_id_3),
                               places=8)
        self.assertAlmostEqual(random_quad_pt_id_3,
                               mat.get_youngs_modulus(random_quad_pt_id_3),
                               places=8)

    def test_young_and_poisson_per_quad_point(self):
        nb_grid_pts = [2, 3, 1]
        lengths = nb_grid_pts
        dim = len(nb_grid_pts)
        form = µ.Formulation.finite_strain
        gradient, weights = µ.linear_finite_elements.gradient_3d
        nb_quad_pts = len(weights)
        cg_tol = 1e-6
        newton_tol = 1e-6
        equil_tol = newton_tol
        maxiter = 100
        verbose = µ.Verbosity.Silent
        young = np.empty(tuple(nb_grid_pts) + (nb_quad_pts, ))
        young[:, :, :, :] = \
            np.arange(1, nb_quad_pts+1)[np.newaxis, np.newaxis, np.newaxis, :]
        poisson = 0.33 * np.ones(tuple(nb_grid_pts) + (nb_quad_pts, ))

        cell = µ.Cell(nb_grid_pts, lengths, form, gradient, weights)
        mat = µ.material.MaterialLinearElastic4_3d.make(cell, "le4")

        # test for error message
        message = lambda name: "Got a wrong shape " + str(nb_quad_pts-1) \
            + "×1 for the " + name + " vector.\n" \
            + "I expected the shape: " + str(nb_quad_pts) + "×1"

        with self.assertRaises(RuntimeError) as context:
            mat.add_pixel(0, young[0, 0, 0, :-1], poisson[0, 0, 0])
        self.assertTrue(message("Youngs modulus")[:78]
                        == str(context.exception)[:78])

        with self.assertRaises(RuntimeError) as context:
            mat.add_pixel(0, young[0, 0, 0], poisson[0, 0, 0, :-1])
        self.assertTrue(message("Poisson ratio")[:77]
                        == str(context.exception)[:77])

        for pixel_index, pixel in enumerate(cell.pixels):
            mat.add_pixel(pixel_index,
                          young[tuple(pixel)], poisson[tuple(pixel)])

        solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)
        DelF = np.array([[0.05, 0.00,  0.00],
                         [0.00, -0.1,  0.00],
                         [0.00, 0.00,  0.02]])
        result = µ.solvers.newton_cg(cell, DelF, solver,
                                     newton_tol, equil_tol, verbose)

        # check if the stress is ordered on each voxel corresponding to the
        # Youngs modulus on the quad points
        def check(P, component, direction):
            P = P[component]
            final = True
            for qpt in range(nb_quad_pts - 1):
                if direction == "increasing":
                    check = (P[qpt] < P[qpt + 1]).all()
                elif direction == "decreasing":
                    check = (P[qpt] > P[qpt + 1]).all()

                final = final and check

            return final

        P = result.stress
        P = P.reshape((dim, dim, nb_quad_pts) + tuple(nb_grid_pts), order='F')

        self.assertTrue(check(P, (0, 0), "increasing"))
        self.assertTrue(check(P, (1, 1), "decreasing"))
        self.assertTrue(check(P, (2, 2), "decreasing"))


if __name__ == '__main__':
    unittest.main()
