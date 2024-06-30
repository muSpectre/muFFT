/**
 * @file   mufft_common.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Jan 2019
 *
 * @brief  Small definitions of commonly used types throughout µFFT
 *
 * Copyright © 2019 Till Junge
 *
 * µFFT is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µFFT is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µFFT; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "libmugrid/grid_common.hh"

#ifndef SRC_LIBMUFFT_MUFFT_COMMON_HH_
#define SRC_LIBMUFFT_MUFFT_COMMON_HH_

namespace muFFT {
  using muGrid::Dim_t;
  using muGrid::Index_t;

  using muGrid::Complex;
  using muGrid::Int;
  using muGrid::Real;
  using muGrid::Uint;

  using muGrid::RuntimeError;

  using muGrid::Ccoord_t;
  using muGrid::DynCcoord_t;
  using muGrid::DynRcoord_t;
  using muGrid::Rcoord_t;
  using muGrid::Shape_t;

  using muGrid::optional;

  using muGrid::oneD;
  using muGrid::threeD;
  using muGrid::twoD;

  using muGrid::OneQuadPt;
  using muGrid::TwoQuadPts;
  using muGrid::FourQuadPts;
  using muGrid::SixQuadPts;
  using muGrid::OneNode;

  using muGrid::Mapping;
  using muGrid::IterUnit;

  /**
   * @enum FFT_PlanFlags
   * @brief Planner flags for FFT.
   * @details This enumeration follows the FFTW library's convention for
   * planning flags. The hope is that this choice will be compatible with
   * alternative FFT implementations.
   */
  enum class FFT_PlanFlags {
    /**
     * @brief Represents the cheapest plan for the slowest execution.
     * @details This flag is used when the priority is to minimize the planning
     * cost, even if it results in slower FFT execution.
     */
    estimate,

    /**
     * @brief Represents a more expensive plan for faster execution.
     * @details This flag is used when the priority is to balance the planning
     * cost and FFT execution speed.
     */
    measure,

    /**
     * @brief Represents the most expensive plan for the fastest execution.
     * @details This flag is used when the priority is to maximize the FFT
     * execution speed, even if it results in higher planning cost.
     */
    patient
  };

  /**
   * @enum FFTDirection
   * @brief Represents the direction of FFT transformation.
   * @details This enum class is used to define the direction of the Fast
   * Fourier Transform (FFT) operation. It defines two possible directions:
   * Forward and Reverse.
   */
  enum class FFTDirection {
    /**
     * @brief Represents the forward direction of FFT transformation.
     * @details This value is used when the FFT operation is to be performed in
     * the forward direction.
     */
    forward,

    /**
     * @brief Represents the reverse direction of FFT transformation.
     * @details This value is used when the FFT operation is to be performed in
     * the reverse direction.
     */
    reverse
  };

  /**
   * @typedef PixelTag
   * @brief A type alias for tagging all fields.
   * @details This type alias is used to tag all fields. The library libµgrid
   * allows for pixel-sub-divisions, which libµFFt does not use.
   */
  using muGrid::PixelTag;

  /**
   * @namespace version
   * @brief A namespace that contains functions related to version information.
   */
  namespace version {

    /**
     * @brief Returns a formatted text that can be printed to stdout or to
     * output files.
     * @details This function returns a formatted text that contains the git
     * commit hash and repository url used to compile µFFT and whether the
     * current state was dirty or not.
     * @return A string that contains the formatted text.
     */
    std::string info();

    /**
     * @brief Returns the git commit hash.
     * @details This function returns the git commit hash used to compile
     * µFFT.
     * @return A constant character pointer that points to the git commit hash.
     */
    const char * hash();

    /**
     * @brief Returns the version string.
     * @details This function returns the formatted string of the version of
     * µFFT.
     * @return A constant character pointer that points to the repository
     * description.
     */
    const char * description();

    /**
     * @brief Checks if the current state was dirty or not.
     * @details This function checks if the current state of the git repository
     * was dirty or not when compiling µFFT.
     * @return A boolean value that indicates whether the current state was
     * dirty or not.
     */
    bool is_dirty();

  }  // namespace version
}  // namespace muFFT

#endif  // SRC_LIBMUFFT_MUFFT_COMMON_HH_
