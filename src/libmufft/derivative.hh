/**
 * @file   derivative.hh
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *         Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   05 June 2019
 *
 * @brief  Representation of finite-differences stencils
 *
 * Copyright © 2019 Lars Pastewka
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

#ifndef SRC_LIBMUFFT_DERIVATIVE_HH_
#define SRC_LIBMUFFT_DERIVATIVE_HH_

#include <memory>

#include "libmugrid/ccoord_operations.hh"
#include "libmugrid/exception.hh"
#include "libmugrid/field_map.hh"
#include "libmugrid/field_typed.hh"
#include "libmugrid/field_collection_global.hh"

#include "mufft_common.hh"

namespace muFFT {
  /**
   * @class DerivativeError
   * @brief A class that represents exceptions related to derivatives.
   * @details This class is derived from the RuntimeError class and is used to
   * handle exceptions related to derivatives.
   */
  class DerivativeError : public RuntimeError {
   public:
    /**
     * @brief A constructor that takes a string as an argument.
     * @param what A string that describes the error.
     */
    explicit DerivativeError(const std::string & what) : RuntimeError(what) {}

    /**
     * @brief A constructor that takes a character array as an argument.
     * @param what A character array that describes the error.
     */
    explicit DerivativeError(const char * what) : RuntimeError(what) {}
  };

  /**
   * @class DerivativeBase
   * @brief A base class that represents a derivative.
   * @details This class provides the basic functionalities for a derivative.
   */
  class DerivativeBase {
   public:
    //! Alias for Eigen::Matrix<Real, Eigen::Dynamic, 1>
    using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    /**
     * @brief Deleted default constructor.
     * @details This constructor is deleted because a DerivativeBase object
     * requires a spatial dimension to be properly initialized.
     */
    DerivativeBase() = delete;

    /**
     * @brief Constructor that takes the spatial dimension as an argument.
     * @param spatial_dimension The spatial dimension of the derivative.
     */
    explicit DerivativeBase(Index_t spatial_dimension);

    //! Default copy constructor
    DerivativeBase(const DerivativeBase & other) = default;

    //! Default move constructor
    DerivativeBase(DerivativeBase && other) = default;

    //! Default destructor
    virtual ~DerivativeBase() = default;

    //! Deleted copy assignment operator
    DerivativeBase & operator=(const DerivativeBase & other) = delete;

    //! Deleted move assignment operator
    DerivativeBase & operator=(DerivativeBase && other) = delete;

    /**
     * @brief A pure virtual function that returns the Fourier representation of
     * the derivative.
     * @param phase The phase is the wavevector times cell dimension, but
     * lacking a factor of 2 π.
     * @return The Fourier representation of the derivative.
     */
    virtual Complex fourier(const Vector & phase) const = 0;

   protected:
    //! The spatial dimension of the problem
    Index_t spatial_dimension;
  };

  /**
   * @class FourierDerivative
   * @brief A class that represents a derivative computed by Fourier
   * interpolation.
   * @details This class is derived from the DerivativeBase class and provides
   * functionalities for Fourier interpolated derivatives.
   */
  class FourierDerivative : public DerivativeBase {
   public:
    //! Alias for the base class
    using Parent = DerivativeBase;

    //! Alias for Eigen::Matrix<Real, Eigen::Dynamic, 1>
    using Vector = typename Parent::Vector;

    /**
     * @brief Deleted default constructor.
     * @details This constructor is deleted because a FourierDerivative object
     * requires a spatial dimension and direction to be properly initialized.
     */
    FourierDerivative() = delete;

    /**
     * @brief Constructor that takes the spatial dimension and direction as
     * arguments.
     * @param spatial_dimension The spatial dimension of the derivative.
     * @param direction The direction of the derivative.
     */
    explicit FourierDerivative(Index_t spatial_dimension, Index_t direction);

    /**
     * @brief Constructor that takes the spatial dimension, direction, and shift
     * info as arguments.
     * @param spatial_dimension The spatial dimension of the derivative.
     * @param direction The direction of the derivative.
     * @param shift The shift information for the derivative.
     */
    explicit FourierDerivative(Index_t spatial_dimension, Index_t direction,
                               const Eigen::ArrayXd & shift);

    //! Default copy constructor
    FourierDerivative(const FourierDerivative & other) = default;

    //! Default move constructor
    FourierDerivative(FourierDerivative && other) = default;

    //! Default destructor
    virtual ~FourierDerivative() = default;

    //! Deleted copy assignment operator
    FourierDerivative & operator=(const FourierDerivative & other) = delete;

    //! Deleted move assignment operator
    FourierDerivative & operator=(FourierDerivative && other) = delete;

    /**
     * @brief Returns the Fourier representation of the Fourier interpolated
     * derivative shifted to the new position of the derivative.
     * @param phase The phase is the wavevector times cell dimension, but
     * lacking a factor of 2 π.
     * @return The Fourier representation of the derivative.
     */
    virtual Complex fourier(const Vector & phase) const;

   protected:
    //! The spatial direction in which to perform differentiation
    Index_t direction;

    //! The real space shift from the position of the center of the cell.
    const Eigen::ArrayXd shift;
  };

  /**
   * @class DiscreteDerivative
   * @brief A class that represents a finite-differences stencil.
   * @details This class is derived from the DerivativeBase class and provides
   * functionalities for finite-differences stencils.
   */
  class DiscreteDerivative : public DerivativeBase {
   public:
    //! Alias for the base class
    using Parent = DerivativeBase;

    //! Alias for Eigen::Matrix<Real, Eigen::Dynamic, 1>
    using Vector = typename Parent::Vector;

    /**
     * @brief Deleted default constructor.
     * @details This constructor is deleted because a DiscreteDerivative object
     * requires stencil information to be properly initialized.
     */
    DiscreteDerivative() = delete;

    /**
     * @brief Constructor with raw stencil information.
     * @param nb_pts The size of the stencil.
     * @param lbounds The relative starting point of the stencil. For example,
     * (-2,) means that the stencil starts two pixels to the left of where the
     * derivative should be computed.
     * @param stencil The coefficients of the stencil.
     */
    DiscreteDerivative(DynCcoord_t nb_pts, DynCcoord_t lbounds,
                       const std::vector<Real> & stencil);

    /**
     * @brief Constructor with raw stencil information.
     * @param nb_pts The size of the stencil.
     * @param lbounds The relative starting point of the stencil.
     * @param stencil The coefficients of the stencil.
     */
    DiscreteDerivative(DynCcoord_t nb_pts, DynCcoord_t lbounds,
                       const Eigen::ArrayXd & stencil);

    //! Default copy constructor
    DiscreteDerivative(const DiscreteDerivative & other) = default;

    //! Default move constructor
    DiscreteDerivative(DiscreteDerivative && other) = default;

    //! Default destructor
    virtual ~DiscreteDerivative() = default;

    //! Deleted copy assignment operator
    DiscreteDerivative & operator=(const DiscreteDerivative & other) = delete;

    //! Deleted move assignment operator
    DiscreteDerivative & operator=(DiscreteDerivative && other) = delete;

    /**
     * @brief Returns the stencil value at a given coordinate.
     * @param dcoord The coordinate.
     * @return The stencil value at the given coordinate.
     */
    Real operator()(const DynCcoord_t & dcoord) const {
      return this->stencil[this->pixels.get_index(dcoord)];
    }

    //! Returns the dimension of the stencil.
    const Dim_t & get_dim() const { return this->pixels.get_dim(); }

    //! Returns the number of grid points in the stencil.
    const DynCcoord_t & get_nb_pts() const {
      return this->pixels.get_nb_subdomain_grid_pts();
    }

    //! Returns the lower bound of the stencil.
    const DynCcoord_t & get_lbounds() const {
      return this->pixels.get_subdomain_locations();
    }

    //! Returns the pixels class that allows to iterate over pixels.
    const muGrid::CcoordOps::DynamicPixels & get_pixels() const {
      return this->pixels;
    }

    /**
     * @brief Applies the "stencil" to a component (degree-of-freedom) of a
     * field and stores the result to a select component of a second field.
     * @details Note that the compiler should have opportunity to inline this
     * function to optimize loops over DOFs.
     * @param in_field The input field.
     * @param in_dof The degree-of-freedom of the input field.
     * @param out_field The output field.
     * @param out_dof The degree-of-freedom of the output field.
     * @param fac A factor to multiply the result with. Defaults to 1.0.
     */
    template <typename T>
    void apply(const muGrid::TypedFieldBase<T> & in_field, Index_t in_dof,
               muGrid::TypedFieldBase<T> & out_field, Index_t out_dof,
               Real fac = 1.0) const;

    /**
     * @brief Returns the Fourier representation of this stencil.
     * @param phase The phase is the wavevector times cell dimension, but
     * lacking a factor of 2 π.
     * @return The Fourier representation of the stencil.
     */
    virtual Complex fourier(const Vector & phase) const;

    /**
     * @brief Returns a new stencil with rolled axes.
     * @details Given a stencil on a three-dimensional grid with axes (x, y, z),
     * the stencil that has been "rolled" by distance one has axes (z, x, y).
     * This is a simple implementation of a rotation operation. For example,
     * given a stencil that described the derivative in the x-direction,
     * rollaxes(1) gives the derivative in the y-direction and rollaxes(2) gives
     * the derivative in the z-direction.
     * @param distance The distance to roll the axes by. Defaults to 1.
     * @return A new stencil with rolled axes.
     */
    DiscreteDerivative rollaxes(int distance = 1) const;

    //! Returns the stencil data.
    const Eigen::ArrayXd & get_stencil() const { return this->stencil; }

   protected:
    //! An object to iterate over the stencil.
    muGrid::CcoordOps::DynamicPixels pixels;

    //! The finite-differences stencil.
    const Eigen::ArrayXd stencil;
  };

  /**
   * @brief Overloads the insertion operator for `muFFT::DiscreteDerivative`.
   * @details This function allows inserting `muFFT::DiscreteDerivative` objects
   * into `std::ostream` objects.
   * @param os The output stream.
   * @param derivative The `muFFT::DiscreteDerivative` object.
   * @return The output stream with the `muFFT::DiscreteDerivative` object
   * inserted.
   */
  std::ostream & operator<<(std::ostream & os,
                            const DiscreteDerivative & derivative);

  //! @brief A convenience alias for a vector of shared pointers to
  //! `DerivativeBase` objects.
  using Gradient_t = std::vector<std::shared_ptr<DerivativeBase>>;

  /**
   * @brief A convenience function to build a gradient operator using exact
   * Fourier differentiation.
   * @details This function creates a gradient operator for a given number of
   * spatial dimensions using Fourier differentiation.
   * @param spatial_dimension The number of spatial dimensions.
   * @return A `Gradient_t` object representing the gradient operator.
   */
  Gradient_t make_fourier_gradient(const Index_t & spatial_dimension);
}  // namespace muFFT

#endif  // SRC_LIBMUFFT_DERIVATIVE_HH_
