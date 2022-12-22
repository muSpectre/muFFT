/**
 * @file   test_solver_newton_cg.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Dec 2017
 *
 * @brief  Tests for the standard Newton-Raphson + Conjugate Gradient solver
 *
 * Copyright © 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "tests.hh"
#include "solver/solvers.hh"
#include "solver/krylov_solver_cg.hh"
#include "solver/krylov_solver_eigen.hh"
#include "projection/projection_finite_strain_fast.hh"
#include "materials/material_linear_elastic1.hh"
#include "cell/cell_factory.hh"

#include <libmugrid/iterators.hh>
#include <libmugrid/ccoord_operations.hh>
#include <libmufft/pocketfft_engine.hh>

#include <boost/mpl/list.hpp>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(newton_cg_tests);

  template <class KrylovSolverType>
  struct KrylovSolverFixture {
    using type = KrylovSolverType;
  };

  using KrylovSolverList =
      boost::mpl::list<KrylovSolverFixture<KrylovSolverCG>,
                       KrylovSolverFixture<KrylovSolverCGEigen>,
                       KrylovSolverFixture<KrylovSolverGMRESEigen>,
                       KrylovSolverFixture<KrylovSolverDGMRESEigen>,
                       KrylovSolverFixture<KrylovSolverBiCGSTABEigen>,
                       KrylovSolverFixture<KrylovSolverMINRESEigen>>;

  BOOST_FIXTURE_TEST_CASE(small_strain_convergence_test,
                          KrylovSolverFixture<KrylovSolverCG>) {
    constexpr Index_t Dim{twoD};
    const DynCcoord_t nb_grid_pts{muGrid::CcoordOps::get_cube<Dim>(Index_t{3})};
    const DynRcoord_t lengths{muGrid::CcoordOps::get_cube<Dim>(1.)};
    constexpr Formulation form{Formulation::small_strain};

    // number of layers in the hard material
    constexpr Index_t nb_lays{1};
    constexpr Real contrast{2};
    if (not(nb_lays < nb_grid_pts[0])) {
      throw std::runtime_error(
          "the number or layers in the hard material must be smaller "
          "than the total number of layers in dimension 0");
    }

    auto cell{make_cell(nb_grid_pts, lengths, form)};

    using Mat_t = MaterialLinearElastic1<Dim>;
    constexpr Real Young{2.}, Poisson{.33};
    auto & material_hard{Mat_t::make(cell, "hard", contrast * Young, Poisson)};
    auto & material_soft{Mat_t::make(cell, "soft", Young, Poisson)};

    for (const auto & pixel_index : cell->get_pixel_indices()) {
      if (pixel_index) {
        material_hard.add_pixel(pixel_index);
      } else {
        material_soft.add_pixel(pixel_index);
      }
    }
    cell->initialise();

    Grad_t<Dim> delEps0{Grad_t<Dim>::Zero()};
    constexpr Real eps0 = 1.;
    // delEps0(0, 1) = delEps0(1, 0) = eps0;
    delEps0(0, 0) = eps0;

    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equil_tol{1e-10};
    constexpr Uint maxiter{Dim * 10};
    constexpr Verbosity verbose{Verbosity::Silent};

    type cg{cell, cg_tol, maxiter, verbose};
    auto result = newton_cg(cell, delEps0, cg, newton_tol, equil_tol, verbose);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(small_strain_patch_dynamic_solver, Fix,
                                   KrylovSolverList, Fix) {
    constexpr Index_t Dim{twoD};
    const Index_t grid_size{3};
    const DynCcoord_t nb_grid_pts{muGrid::CcoordOps::get_cube<Dim>(grid_size)};
    const DynRcoord_t lengths{muGrid::CcoordOps::get_cube<Dim>(1.)};
    constexpr Formulation form{Formulation::small_strain};

    // number of layers in the hard material
    constexpr Index_t nb_lays{1};
    constexpr Real contrast{2};
    if (not(nb_lays < nb_grid_pts[0])) {
      throw std::runtime_error(
          "the number or layers in the hard material must be smaller "
          "than the total number of layers in dimension 0");
    }

    auto cell{make_cell(nb_grid_pts, lengths, form)};

    using Mat_t = MaterialLinearElastic1<Dim>;
    constexpr Real Young{2.}, Poisson{.33};
    auto & material_hard{Mat_t::make(cell, "hard", contrast * Young, Poisson)};
    auto & material_soft{Mat_t::make(cell, "soft", Young, Poisson)};

    for (const auto && index_pixel : akantu::enumerate(cell->get_pixels())) {
      auto && index{std::get<0>(index_pixel)};
      auto && pixel{std::get<1>(index_pixel)};
      if (pixel[0] < Index_t(nb_lays)) {
        material_hard.add_pixel(index);
      } else {
        material_soft.add_pixel(index);
      }
    }
    cell->initialise();

    Grad_t<Dim> delEps0{Grad_t<Dim>::Zero()};
    constexpr Real eps0 = 1.;
    // delEps0(0, 1) = delEps0(1, 0) = eps0;
    delEps0(0, 0) = eps0;

    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equil_tol{1e-10};
    constexpr Uint maxiter{Dim * 10};
    constexpr Verbosity verbose{Verbosity::Silent};

    using KrylovSolver_t = typename Fix::type;
    KrylovSolver_t cg{cell, cg_tol, maxiter, verbose};
    auto result = newton_cg(cell, delEps0, cg, newton_tol, equil_tol, verbose);
    if (verbose > Verbosity::Silent) {
      std::cout << "result:" << std::endl << result.grad << std::endl;
      std::cout << "mean strain = " << std::endl
                << cell->get_strain().get_sub_pt_map().mean() << std::endl;
    }

    /**
     *  verification of resultant strains: subscript ₕ for hard and ₛ
     *  for soft, Nₕ is nb_lays and Nₜₒₜ is nb_grid_pts, k is contrast
     *
     *     Δl = εl = Δlₕ + Δlₛ = εₕlₕ+εₛlₛ
     *  => ε = εₕ Nₕ/Nₜₒₜ + εₛ (Nₜₒₜ-Nₕ)/Nₜₒₜ
     *
     *  σ is constant across all layers
     *        σₕ = σₛ
     *  => Eₕ εₕ = Eₛ εₛ
     *  => εₕ = 1/k εₛ
     *  => ε / (1/k Nₕ/Nₜₒₜ + (Nₜₒₜ-Nₕ)/Nₜₒₜ) = εₛ
     */
    const Real factor{1 / contrast * Real(nb_lays) / nb_grid_pts[0] + 1. -
                      nb_lays / Real(nb_grid_pts[0])};
    const Real eps_soft{eps0 / factor};
    const Real eps_hard{eps_soft / contrast};
    if (verbose > Verbosity::Silent) {
      std::cout << "εₕ = " << eps_hard << ", εₛ = " << eps_soft << std::endl;
      std::cout << "ε = εₕ Nₕ/Nₜₒₜ + εₛ (Nₜₒₜ-Nₕ)/Nₜₒₜ" << std::endl;
    }
    Grad_t<Dim> Eps_hard;
    Eps_hard << eps_hard, 0, 0, 0;
    Grad_t<Dim> Eps_soft;
    Eps_soft << eps_soft, 0, 0, 0;

    // verify uniaxial tension patch test
    for (const auto & index_pixel :
         akantu::zip(cell->get_pixel_indices(), cell->get_pixels())) {
      auto && index{std::get<0>(index_pixel)};
      auto && pixel{std::get<1>(index_pixel)};
      if (pixel[0] < Index_t(nb_lays)) {
        BOOST_CHECK_LE(
            (Eps_hard - cell->get_strain().get_pixel_map(Dim)[index]).norm(),
            tol);
      } else {
        BOOST_CHECK_LE(
            (Eps_soft - cell->get_strain().get_pixel_map(Dim)[index]).norm(),
            tol);
      }
    }

    delEps0 = Grad_t<Dim>::Zero();
    delEps0(0, 1) = delEps0(1, 0) = eps0;

    KrylovSolver_t cg2{cell, cg_tol, maxiter, verbose};
    result = de_geus(cell, delEps0, cg2, newton_tol, equil_tol, verbose);
    Eps_hard << 0, eps_hard, eps_hard, 0;
    Eps_soft << 0, eps_soft, eps_soft, 0;

    // verify pure shear patch test
    for (const auto & index_pixel :
         akantu::zip(cell->get_pixel_indices(), cell->get_pixels())) {
      auto && index{std::get<0>(index_pixel)};
      auto && pixel{std::get<1>(index_pixel)};
      auto && strain = cell->get_strain().get_pixel_map(Dim)[index];
      if (pixel[0] < Index_t(nb_lays)) {
        BOOST_CHECK_LE((Eps_hard - strain).norm(), tol);
      } else {
        BOOST_CHECK_LE((Eps_soft - strain).norm(), tol);
      }
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(finite_strain_patch_dynamic_solver, Fix,
                                   KrylovSolverList, Fix) {
    constexpr Index_t Dim{twoD};
    constexpr Index_t GridSize{3};
    const DynCcoord_t nb_grid_pts{muGrid::CcoordOps::get_cube<Dim>(GridSize)};
    const DynRcoord_t lengths{muGrid::CcoordOps::get_cube<Dim>(1.)};
    constexpr Formulation form{Formulation::finite_strain};
    // because we compare finitie strain results to small strain predictions
    constexpr Real loose_tol(1e-6);

    // number of layers in the hard material
    constexpr Index_t nb_lays{1};
    constexpr Real contrast{2};
    if (not(nb_lays < nb_grid_pts[0])) {
      throw std::runtime_error(
          "the number or layers in the hard material must be smaller "
          "than the total number of layers in dimension 0");
    }

    auto cell{make_cell(nb_grid_pts, lengths, form)};

    using Mat_t = MaterialLinearElastic1<Dim>;
    constexpr Real Young{2.}, Poisson{.33};
    auto & material_hard{Mat_t::make(cell, "hard", contrast * Young, Poisson)};
    auto & material_soft{Mat_t::make(cell, "soft", Young, Poisson)};

    for (const auto && index_pixel : cell->get_pixels().enumerate()) {
      auto && index{std::get<0>(index_pixel)};
      auto && pixel{std::get<1>(index_pixel)};
      if (pixel[0] < Index_t(nb_lays)) {
        material_hard.add_pixel(index);
      } else {
        material_soft.add_pixel(index);
      }
    }
    cell->initialise();

    Grad_t<Dim> delEps0{Grad_t<Dim>::Zero()};
    constexpr Real eps0 = 1.e-4;
    // delEps0(0, 1) = delEps0(1, 0) = eps0;
    delEps0(0, 0) = eps0;

    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equil_tol{1e-10};
    constexpr Uint maxiter{Dim * 10};
    constexpr Verbosity verbose{Verbosity::Silent};

    using KrylovSolver_t = typename Fix::type;
    KrylovSolver_t cg{cell, cg_tol, maxiter, verbose};
    auto result = newton_cg(cell, delEps0, cg, newton_tol, equil_tol, verbose);
    if (verbose > Verbosity::Silent) {
      std::cout << "result:" << std::endl << result.grad << std::endl;
      std::cout << "mean strain = " << std::endl
                << cell->get_strain().get_sub_pt_map().mean() << std::endl;
    }

    /**
     *  verification of resultant strains: subscript ₕ for hard and ₛ
     *  for soft, Nₕ is nb_lays and Nₜₒₜ is nb_grid_pts, k is contrast
     *
     *     Δl = εl = Δlₕ + Δlₛ = εₕlₕ+εₛlₛ
     *  => ε = εₕ Nₕ/Nₜₒₜ + εₛ (Nₜₒₜ-Nₕ)/Nₜₒₜ
     *
     *  σ is constant across all layers
     *        σₕ = σₛ
     *  => Eₕ εₕ = Eₛ εₛ
     *  => εₕ = 1/k εₛ
     *  => ε / (1/k Nₕ/Nₜₒₜ + (Nₜₒₜ-Nₕ)/Nₜₒₜ) = εₛ
     */
    const Real factor{1 / contrast * Real(nb_lays) / nb_grid_pts[0] + 1. -
                      nb_lays / Real(nb_grid_pts[0])};
    const Real eps_soft{eps0 / factor};
    const Real eps_hard{eps_soft / contrast};
    if (verbose > Verbosity::Silent) {
      std::cout << "εₕ = " << eps_hard << ", εₛ = " << eps_soft << std::endl;
      std::cout << "ε = εₕ Nₕ/Nₜₒₜ + εₛ (Nₜₒₜ-Nₕ)/Nₜₒₜ" << std::endl;
    }
    Grad_t<Dim> Eps_hard;
    Eps_hard << eps_hard, 0, 0, 0;
    Grad_t<Dim> Eps_soft;
    Eps_soft << eps_soft, 0, 0, 0;

    // verify uniaxial tension patch test
    for (const auto & index_pixel :
         akantu::zip(cell->get_pixel_indices(), cell->get_pixels())) {
      auto && index{std::get<0>(index_pixel)};
      auto && pixel{std::get<1>(index_pixel)};
      if (pixel[0] < Index_t(nb_lays)) {
        BOOST_CHECK_LE((Eps_hard + Eps_hard.Identity() -
                        cell->get_strain().get_pixel_map(Dim)[index])
                           .norm(),
                       loose_tol);
      } else {
        BOOST_CHECK_LE((Eps_soft + Eps_hard.Identity() -
                        cell->get_strain().get_pixel_map(Dim)[index])
                           .norm(),
                       loose_tol);
      }
    }

    delEps0 = Grad_t<Dim>::Zero();
    delEps0(0, 1) = delEps0(1, 0) = eps0;

    KrylovSolver_t cg2{cell, cg_tol, maxiter, verbose};
    result = de_geus(cell, delEps0, cg2, newton_tol, equil_tol, verbose);
    Eps_hard << 0, eps_hard, eps_hard, 0;
    Eps_soft << 0, eps_soft, eps_soft, 0;

    // verify pure shear patch test
    for (const auto & index_pixel :
         akantu::zip(cell->get_pixel_indices(), cell->get_pixels())) {
      auto && index{std::get<0>(index_pixel)};
      auto && pixel{std::get<1>(index_pixel)};
      auto strain{cell->get_strain().get_pixel_map(Dim)[index]};
      Eigen::Matrix<Real, Dim, Dim> E{
          .5 * (strain.transpose() * strain - strain.Identity(Dim, Dim))};
      Real error{};
      if (pixel[0] < Index_t(nb_lays)) {
        error = (Eps_hard - E).norm();
      } else {
        error = (Eps_soft - E).norm();
      }
      BOOST_CHECK_LE(error, loose_tol);
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
