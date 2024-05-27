/**
 * @file   fft_utils.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   06 Dec 2017
 *
 * @brief  collection of functions used in the context of spectral operations
 *
 * Copyright © 2017 Till Junge
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

#ifndef SRC_LIBMUFFT_FFT_UTILS_HH_
#define SRC_LIBMUFFT_FFT_UTILS_HH_

#include "libmugrid/exception.hh"

#include "mufft_common.hh"

#include "Eigen/Dense"

#include <array>
#include <valarray>

namespace muFFT {

  namespace internal {
    //! computes hermitian size according to FFTW
    template <Dim_t Dim, size_t... I>
    constexpr Ccoord_t<Dim> herm(const Ccoord_t<Dim> & nb_grid_pts,
                                 std::index_sequence<I...>) {
      return Ccoord_t<Dim>{nb_grid_pts.front() / 2 + 1, nb_grid_pts[I + 1]...};
    }
  }  // namespace internal

  /**
   * @brief Returns the hermitian grid corresponding to a full grid.
   *
   * This function template returns the hermitian grid that corresponds to a
   * real-valued field on a full grid, assuming that the last dimension is not
   * fully represented in reciprocal space because of symmetry.
   *
   * @tparam dim The dimensionality of the grid.
   * @param full_nb_grid_pts A Ccoord_t object representing the number of grid
   * points in the full grid.
   * @return A Ccoord_t object representing the hermitian grid.
   */
  template <size_t dim>
  constexpr Ccoord_t<dim>
  get_nb_hermitian_grid_pts(Ccoord_t<dim> full_nb_grid_pts) {
    return internal::herm<dim>(full_nb_grid_pts,
                               std::make_index_sequence<dim - 1>{});
  }

  /**
   * @brief Returns the hermitian grid corresponding to a full grid.
   *
   * This function template returns the hermitian grid that corresponds to a
   * real-valued field on a full grid, assuming that the last dimension is not
   * fully represented in reciprocal space because of symmetry. It supports
   * dynamic dimensionality.
   *
   * @tparam MaxDim The maximum dimensionality of the grid.
   * @param full_nb_grid_pts A muGrid::DynCcoord object representing the number
   * of grid points in the full grid.
   * @return A muGrid::DynCcoord object representing the hermitian grid.
   * @throws RuntimeError if the dimensionality of the grid is not 1, 2, or 3.
   */
  template <size_t MaxDim>
  inline muGrid::DynCcoord<MaxDim>
  get_nb_hermitian_grid_pts(muGrid::DynCcoord<MaxDim> full_nb_grid_pts) {
    switch (full_nb_grid_pts.get_dim()) {
    case oneD: {
      return muGrid::DynCcoord<MaxDim>{
          get_nb_hermitian_grid_pts(Ccoord_t<oneD>(full_nb_grid_pts))};
      break;
    }
    case twoD: {
      return muGrid::DynCcoord<MaxDim>{
          get_nb_hermitian_grid_pts(Ccoord_t<twoD>(full_nb_grid_pts))};
      break;
    }
    case threeD: {
      return muGrid::DynCcoord<MaxDim>{
          get_nb_hermitian_grid_pts(Ccoord_t<threeD>(full_nb_grid_pts))};
      break;
    }
    default:
      throw RuntimeError("Only 1, 2, and 3-dimensional cases are allowed");
      break;
    }
  }

  /**
   * @brief Compute nondimensional FFT frequency.
   *
   * This function computes the FFT frequency for a given index and number of
   * samples. The frequency is normalized to the number of samples.
   *
   * @param i The index for which to compute the FFT frequency.
   * @param nb_samples The number of samples.
   * @return The FFT frequency for the given index and number of samples.
   */
  inline Int fft_freq(Int i, size_t nb_samples) {
    Int N = (nb_samples - 1) / 2 + 1;  // needs to be signed int for neg freqs
    if (i < N) {
      return i;
    } else {
      return -Int(nb_samples) / 2 + i - N;
    }
  }

  /**
   * @brief Compute FFT frequency in correct length or time units.
   *
   * This function computes the FFT frequency for a given index and number of
   * samples, taking into account the correct length or time units. The length
   * refers to the total size of the domain over which the FFT is taken (for
   * instance the length of an edge of an RVE).
   *
   * @param i The index for which to compute the FFT frequency.
   * @param nb_samples The number of samples.
   * @param length The length of the domain over which the FFT is taken.
   * @return The FFT frequency for the given index and number of samples, in the
   * correct length or time units.
   */
  inline Real fft_freq(Int i, size_t nb_samples, Real length) {
    return static_cast<Real>(fft_freq(i, nb_samples)) / length;
  }

  /**
   * @brief Compute nondimensional FFT frequencies.
   *
   * This function computes the FFT frequencies for a given index and number of
   * samples. The frequency is normalized to the number of samples.
   *
   * @param nb_samples The number of samples.
   * @return A valarray containing the FFT frequencies for the given number of
   * samples.
   */
  std::valarray<Real> fft_freqs(size_t nb_samples);

  /**
   * @brief Compute FFT frequencies in correct length or time units.
   *
   * This function computes the FFT frequencies for a given number of samples,
   * taking into account the correct length or time units. The length refers to
   * the total size of the domain over which the FFT is taken (for instance the
   * length of an edge of an RVE).
   *
   * @param nb_samples The number of samples.
   * @param length The length of the domain over which the FFT is taken.
   * @return A valarray containing the FFT frequencies for the given number of
   * samples, in the correct length or time units.
   */
  std::valarray<Real> fft_freqs(size_t nb_samples, Real length);

  /**
   * @brief Compute multidimensional nondimensional FFT frequencies.
   *
   * This function computes the FFT frequencies for a given index and number of
   * samples for multidimensional fields. The frequency is normalized to the
   * number of samples.
   *
   * @tparam dim The dimensionality of the grid.
   * @param nb_grid_pts A Ccoord_t object representing the number of grid points
   * in each dimension.
   * @return An array of valarrays where each valarray contains the FFT
   * frequencies for a specific dimension.
   */
  template <size_t dim>
  inline std::array<std::valarray<Real>, dim>
  fft_freqs(Ccoord_t<dim> nb_grid_pts) {
    std::array<std::valarray<Real>, dim> retval{};
    for (size_t i = 0; i < dim; ++i) {
      retval[i] = std::move(fft_freqs(nb_grid_pts[i]));
    }
    return retval;
  }

  /**
   * @brief Compute multidimensional FFT frequencies for a grid in correct
   * length or time units.
   *
   * This function template computes the FFT frequencies for a grid, taking into
   * account the correct length or time units. It iterates over each dimension
   * of the grid, computes the FFT frequencies for that dimension, and stores
   * them in a return array.
   *
   * @tparam dim The dimensionality of the grid.
   * @param nb_grid_pts A Ccoord_t object representing the number of grid points
   * in each dimension.
   * @param lengths An array representing the lengths of the domain in each
   * dimension.
   * @return An array of valarrays where each valarray contains the FFT
   * frequencies for a specific dimension.
   */
  template <size_t dim>
  inline std::array<std::valarray<Real>, dim>
  fft_freqs(Ccoord_t<dim> nb_grid_pts, std::array<Real, dim> lengths) {
    std::array<std::valarray<Real>, dim> retval{};
    for (size_t i = 0; i < dim; ++i) {
      retval[i] = std::move(fft_freqs(nb_grid_pts[i], lengths[i]));
    }
    return retval;
  }

  /**
   * @brief A class that encapsulates the creation and retrieval of wave
   * vectors.
   *
   * This class is templated on the dimensionality of the wave vectors. It
   * provides methods to get unnormalized, normalized, and complex wave vectors.
   * It also provides methods to get the number of grid points in a specific
   * dimension.
   *
   * @tparam dim The dimensionality of the wave vectors.
   */
  template <Dim_t dim>
  class FFT_freqs {
   public:
    /**
     * @brief A type alias for an Eigen matrix with dimensions equivalent to
     * Ccoord_t.
     */
    using CcoordVector = Eigen::Matrix<Dim_t, dim, 1>;

    /**
     * @brief A type alias for an Eigen matrix representing a wave vector.
     */
    using Vector = Eigen::Matrix<Real, dim, 1>;

    /**
     * @brief A type alias for an Eigen matrix representing a complex wave
     * vector.
     */
    using VectorComplex = Eigen::Matrix<Complex, dim, 1>;

    /**
     * @brief Default constructor is deleted to prevent creating an object
     * without specifying the grid points.
     */
    FFT_freqs() = delete;

    /**
     * @brief Constructor that initializes the object with the number of grid
     * points.
     *
     * @param nb_grid_pts The number of grid points in each dimension.
     */
    explicit FFT_freqs(Ccoord_t<dim> nb_grid_pts)
        : freqs{fft_freqs(nb_grid_pts)} {}

    /**
     * @brief Constructor that initializes the object with the number of grid
     * points and the lengths of the domain.
     *
     * @param nb_grid_pts The number of grid points in each dimension.
     * @param lengths The lengths of the domain in each dimension.
     */
    FFT_freqs(Ccoord_t<dim> nb_grid_pts, std::array<Real, dim> lengths)
        : freqs{fft_freqs(nb_grid_pts, lengths)} {}

    /**
     * @brief Copy constructor is deleted to prevent copying of the object.
     */
    FFT_freqs(const FFT_freqs & other) = delete;

    /**
     * @brief Move constructor.
     */
    FFT_freqs(FFT_freqs && other) = default;

    /**
     * @brief Destructor.
     */
    virtual ~FFT_freqs() = default;

    /**
     * @brief Copy assignment operator is deleted to prevent copying of the
     * object.
     */
    FFT_freqs & operator=(const FFT_freqs & other) = delete;

    /**
     * @brief Move assignment operator.
     */
    FFT_freqs & operator=(FFT_freqs && other) = default;

    /**
     * @brief Get the unnormalized wave vector in sampling units.
     *
     * @param ccoord The coordinates in the frequency domain.
     * @return The unnormalized wave vector.
     */
    inline Vector get_xi(const Ccoord_t<dim> ccoord) const;

    /**
     * @brief Get the unnormalized complex wave vector in sampling units.
     *
     * @param ccoord The coordinates in the frequency domain.
     * @return The unnormalized complex wave vector.
     */
    inline VectorComplex get_complex_xi(const Ccoord_t<dim> ccoord) const;

    /**
     * @brief Get the normalized wave vector.
     *
     * @param ccoord The coordinates in the frequency domain.
     * @return The normalized wave vector.
     */
    inline Vector get_unit_xi(const Ccoord_t<dim> ccoord) const {
      auto && xi = this->get_xi(std::move(ccoord));
      return xi / xi.norm();
    }

    /**
     * @brief Get the number of grid points in a specific dimension.
     *
     * @param i The index of the dimension.
     * @return The number of grid points in the specified dimension.
     */
    inline Index_t get_nb_grid_pts(Index_t i) const {
      return this->freqs[i].size();
    }

   protected:
    /**
     * @brief A container for frequencies ordered by spatial dimension.
     */
    const std::array<std::valarray<Real>, dim> freqs;
  };

  /**
   * @brief Get the xi value for the FFT frequencies.
   *
   * This function is a member of the FFT_freqs class template. It takes a
   * constant Ccoord_t object as an argument, which represents the coordinates
   * in the frequency domain. The function iterates over each dimension and
   * assigns the corresponding frequency to the return vector.
   *
   * @tparam dim The dimensionality of the FFT operation.
   * @param ccoord A constant reference to a Ccoord_t object representing the
   * coordinates in the frequency domain.
   * @return A Vector object where each element is the frequency corresponding
   * to the coordinate in the same dimension.
   */
  template <Dim_t dim>
  typename FFT_freqs<dim>::Vector
  FFT_freqs<dim>::get_xi(const Ccoord_t<dim> ccoord) const {
    Vector retval{};
    for (Index_t i = 0; i < dim; ++i) {
      retval(i) = this->freqs[i][ccoord[i]];
    }
    return retval;
  }
}  // namespace muFFT

#endif  // SRC_LIBMUFFT_FFT_UTILS_HH_
