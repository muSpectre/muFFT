/**
 * @file   cell_factory.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   15 Dec 2017
 *
 * @brief  Cell factories to help create cells with ease
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

#ifndef SRC_CELL_CELL_FACTORY_HH_
#define SRC_CELL_CELL_FACTORY_HH_

#include <memory>

#include <libmugrid/ccoord_operations.hh>
#include <libmugrid/communicator.hh>

#include <libmufft/derivative.hh>
#include <libmufft/pocketfft_engine.hh>

#include "common/muSpectre_common.hh"
#include "cell/cell.hh"
#include "projection/projection_finite_strain_fast.hh"
#include "projection/projection_small_strain.hh"

using muGrid::RuntimeError;

namespace muSpectre {

  namespace internal {

    /**
     * function to create consistent input for the constructor of
     * `muSpectre::Cell`. Users should never need to call this function, for
     * internal use only
     */
    template <size_t DimS, class FFTEngine>
    inline std::unique_ptr<ProjectionBase> cell_input_helper(
        const DynCcoord_t & nb_grid_pts, const DynRcoord_t & lengths,
        const Formulation & form,
        ProjectionBase::Gradient_t gradient, ProjectionBase::Weights_t weights,
        const muFFT::Communicator & comm = muFFT::Communicator(),
        const muFFT::FFT_PlanFlags & flags = muFFT::FFT_PlanFlags::estimate) {
      auto && dim{nb_grid_pts.get_dim()};
      if (static_cast<Index_t>(gradient.size()) % dim != 0) {
        std::stringstream error{};
        error << "There are " << gradient.size() << " derivative operators in "
              << "the gradient. This number must be divisible by the system "
              << "dimension " << dim << ".";
        throw RuntimeError(error.str());
      }
      // Deduce number of quad points from the gradient
      const auto nb_quad_pts{gradient.size() / dim};
      auto fft_ptr{std::make_unique<FFTEngine>(nb_grid_pts, comm, flags)};
      switch (form) {
      case Formulation::finite_strain: {
        if (nb_quad_pts == OneQuadPt) {
          using Projection = ProjectionFiniteStrainFast<DimS, OneQuadPt>;
          return std::make_unique<Projection>(std::move(fft_ptr), lengths,
                                              gradient, weights);
        } else if (nb_quad_pts == TwoQuadPts) {
          using Projection = ProjectionFiniteStrainFast<DimS, TwoQuadPts>;
          return std::make_unique<Projection>(std::move(fft_ptr), lengths,
                                              gradient, weights);
        } else if (nb_quad_pts == FourQuadPts) {
          using Projection = ProjectionFiniteStrainFast<DimS, FourQuadPts>;
          return std::make_unique<Projection>(std::move(fft_ptr), lengths,
                                              gradient, weights);
        } else if (nb_quad_pts == FiveQuadPts) {
          using Projection = ProjectionFiniteStrainFast<DimS, FiveQuadPts>;
          return std::make_unique<Projection>(std::move(fft_ptr), lengths,
                                              gradient, weights);
        } else if (nb_quad_pts == SixQuadPts) {
          using Projection = ProjectionFiniteStrainFast<DimS, SixQuadPts>;
          return std::make_unique<Projection>(std::move(fft_ptr), lengths,
                                              gradient, weights);
        } else {
          std::stringstream error;
          error << nb_quad_pts << " quadrature points are presently "
                << "unsupported for finite strain calculations.";
          throw RuntimeError(error.str());
        }
      }
      case Formulation::small_strain: {
        if (nb_quad_pts == OneQuadPt) {
          using Projection = ProjectionSmallStrain<DimS, OneQuadPt>;
          return std::make_unique<Projection>(std::move(fft_ptr), lengths,
                                              gradient, weights);
        } else if (nb_quad_pts == TwoQuadPts) {
          using Projection = ProjectionSmallStrain<DimS, TwoQuadPts>;
          return std::make_unique<Projection>(std::move(fft_ptr), lengths,
                                              gradient, weights);
        } else if (nb_quad_pts == FourQuadPts) {
          using Projection = ProjectionSmallStrain<DimS, FourQuadPts>;
          return std::make_unique<Projection>(std::move(fft_ptr), lengths,
                                              gradient, weights);
        } else if (nb_quad_pts == FiveQuadPts) {
          using Projection = ProjectionSmallStrain<DimS, FiveQuadPts>;
          return std::make_unique<Projection>(std::move(fft_ptr), lengths,
                                              gradient, weights);
        } else if (nb_quad_pts == SixQuadPts) {
          using Projection = ProjectionSmallStrain<DimS, SixQuadPts>;
          return std::make_unique<Projection>(std::move(fft_ptr), lengths,
                                              gradient, weights);
        } else {
          std::stringstream error;
          error << nb_quad_pts << " quadrature points are presently "
                << "unsupported for small strain calculations.";
          throw RuntimeError(error.str());
        }
      }
      default: {
        throw RuntimeError("Unknown formulation.");
        break;
      }
      }
      throw RuntimeError("Internal error: At end of cell_input_helper");
      return nullptr;  // required by g++5.4 in debug mode only
    }

  }  // namespace internal

  /**
   * Convenience function to create consistent input for the constructor of *
   * `muSpectre::Cell`. Creates a unique ptr to a Projection operator (with
   * appropriate FFT_engine) to be used in a cell constructor
   *
   * @param nb_grid_pts resolution of the discretisation grid in each spatial
   * directional
   * @param lengths length of the computational domain in each spatial direction
   * @param form problem formulation (small vs finite strain)
   * @param gradient gradient operator to use (i.e., "exact" Fourier derivation,
   * finite differences, etc)
   * @param comm communicator used for solving distributed problems
   */
  template <class FFTEngine = muFFT::PocketFFTEngine>
  inline std::unique_ptr<ProjectionBase> cell_input(
      const DynCcoord_t & nb_grid_pts, const DynRcoord_t & lengths,
      const Formulation & form,
      ProjectionBase::Gradient_t gradient, ProjectionBase::Weights_t weights,
      const muFFT::Communicator & comm = muFFT::Communicator(),
      const muFFT::FFT_PlanFlags & flags = muFFT::FFT_PlanFlags::estimate) {
    const Index_t dim{nb_grid_pts.get_dim()};
    if (dim != lengths.get_dim()) {
      std::stringstream error{};
      error << "Dimension mismatch between nb_grid_pts (dim = " << dim
            << ") and lengths (dim = " << lengths.get_dim() << ").";
      throw RuntimeError(error.str());
    }
    switch (dim) {
    case oneD: {
      return internal::cell_input_helper<oneD, FFTEngine>(
          nb_grid_pts, lengths, form, gradient, weights, comm, flags);
      break;
    }
    case twoD: {
      return internal::cell_input_helper<twoD, FFTEngine>(
          nb_grid_pts, lengths, form, gradient, weights, comm, flags);
      break;
    }
    case threeD: {
      return internal::cell_input_helper<threeD, FFTEngine>(
          nb_grid_pts, lengths, form, gradient, weights, comm, flags);
      break;
    }
    default:
      throw RuntimeError("Unknown dimension.");
      break;
    }
    throw RuntimeError("Internal error: At end of cell_input");
  }

  /**
   * Convenience function to create consistent input for the constructor of *
   * `muSpectre::Cell`. Creates a unique ptr to a Projection operator (with
   * appropriate FFT_engine) to be used in a cell constructor. Uses the "exact"
   * fourier derivation operator for calculating gradients
   *
   * @param nb_grid_pts resolution of the discretisation grid in each spatial
   * directional
   * @param lengths length of the computational domain in each spatial direction
   * @param form problem formulation (small vs finite strain)
   * @param comm communicator used for solving distributed problems
   */
  template <class FFTEngine = muFFT::PocketFFTEngine>
  inline std::unique_ptr<ProjectionBase> cell_input(
      const DynCcoord_t & nb_grid_pts, const DynRcoord_t & lengths,
      const Formulation & form,
      const muFFT::Communicator & comm = muFFT::Communicator(),
      const muFFT::FFT_PlanFlags & flags = muFFT::FFT_PlanFlags::estimate) {
    const Index_t dim{nb_grid_pts.get_dim()};
    return cell_input<FFTEngine>(nb_grid_pts, lengths, form,
                                 muFFT::make_fourier_gradient(dim), {1},
                                 comm, flags);
  }

  /**
   * convenience function to create a cell (avoids having to build
   * and move the chain of unique_ptrs
   *
   * @param nb_grid_pts resolution of the discretisation grid in each spatial
   * directional
   * @param lengths length of the computational domain in each spatial direction
   * @param form problem formulation (small vs finite strain)
   * @param gradient gradient operator to use (i.e., "exact" Fourier derivation,
   * finite differences, etc)
   * @param comm communicator used for solving distributed problems
   */
  template <typename Cell_t = Cell, class FFTEngine = muFFT::PocketFFTEngine>
  inline std::shared_ptr<Cell_t> make_cell(
      DynCcoord_t nb_grid_pts, DynRcoord_t lengths, Formulation form,
      ProjectionBase::Gradient_t gradient, ProjectionBase::Weights_t weights,
      const muFFT::Communicator & comm = muFFT::Communicator(),
      const muFFT::FFT_PlanFlags & flags = muFFT::FFT_PlanFlags::estimate) {
    return std::make_shared<Cell_t>(
        cell_input<FFTEngine>(nb_grid_pts, lengths, form, gradient, weights,
                              comm, flags));
  }

  /**
   * convenience function to create a cell (avoids having to build
   * and move the chain of unique_ptrs. Uses the "exact" fourier derivation
   * operator for calculating gradients
   *
   * @param nb_grid_pts resolution of the discretisation grid in each spatial
   * directional
   * @param lengths length of the computational domain in each spatial direction
   * @param form problem formulation (small vs finite strain)
   * @param comm communicator used for solving distributed problems
   */
  template <typename Cell_t = Cell, class FFTEngine = muFFT::PocketFFTEngine>
  inline std::shared_ptr<Cell_t> make_cell(
      DynCcoord_t nb_grid_pts, DynRcoord_t lengths, Formulation form,
      const muFFT::Communicator & comm = muFFT::Communicator(),
      const muFFT::FFT_PlanFlags & flags = muFFT::FFT_PlanFlags::estimate) {
    const Index_t dim{nb_grid_pts.get_dim()};
    return make_cell<Cell_t, FFTEngine>(nb_grid_pts, lengths, form,
                                        muFFT::make_fourier_gradient(dim), {1},
                                        comm,
                                        flags);
  }

  /**
   * convenience function to create a cell with default communicator (avoids
   * having to build and move the chain of unique_ptrs. Uses the "exact" fourier
   * derivation operator for calculating gradients
   *
   * @param nb_grid_pts resolution of the discretisation grid in each spatial
   * directional
   * @param lengths length of the computational domain in each spatial direction
   * @param form problem formulation (small vs finite strain)
   * @param comm communicator used for solving distributed problems
   */
  template <typename Cell_t = Cell, class FFTEngine = muFFT::PocketFFTEngine>
  inline std::shared_ptr<Cell_t> make_cell(
      DynCcoord_t nb_grid_pts, DynRcoord_t lengths, Formulation form,
      const muFFT::FFT_PlanFlags & flags) {
    const Index_t dim{nb_grid_pts.get_dim()};
    return make_cell<Cell_t, FFTEngine>(nb_grid_pts, lengths, form,
                                        muFFT::make_fourier_gradient(dim),
                                        {1},
                                        muFFT::Communicator(), flags);
  }

}  // namespace muSpectre

#endif  // SRC_CELL_CELL_FACTORY_HH_
