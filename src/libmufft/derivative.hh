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
     * @brief Returns the Fourier derivative.
     * @details Returns the derivative of the Fourier interpolated field.
     * @param phase The phase is the wavevector times cell dimension, but
     * lacking a factor of 2 π.
     * @return The Fourier representation of the derivative.
     */
    virtual Complex fourier(const Vector & phase) const {
      return Complex(0, 2 * muGrid::pi * phase[this->direction]) *
             std::exp(
                 Complex(0, 2 * muGrid::pi * this->shift.matrix().dot(phase)));
    }

   protected:
    //! The spatial direction in which to perform differentiation
    Index_t direction;

    //! The real space shift from the position of the center of the cell.
    const Eigen::ArrayXd shift;
  };

  /**
   * @class DiscreteDerivative
   * @brief A class that represents a discrete derivative.
   * @details This class is derived from the DerivativeBase class and provides
   * functionalities for discrete derivatives.
   */
  class DiscreteDerivative : public DerivativeBase {
   public:
    using Parent = DerivativeBase;  //!< Base class alias
    using Vector =
        typename Parent::Vector;  //!< Vector type alias from the base class

    //! Default constructor is deleted as a DiscreteDerivative object requires
    //! raw stencil information for proper initialization
    DiscreteDerivative() = delete;

    /**
     * @brief Constructor with raw stencil information
     * @details This constructor initializes a DiscreteDerivative object with
     * the provided stencil size, relative starting point of stencil, and
     * stencil coefficients.
     * @param nb_pts The stencil size.
     * @param lbounds The relative starting point of stencil. For example, a
     * value of (-2,) means that the stencil starts two pixels to the left of
     * where the derivative should be computed.
     * @param stencil The stencil coefficients.
     */
    DiscreteDerivative(DynCcoord_t nb_pts, DynCcoord_t lbounds,
                       const std::vector<Real> & stencil);

    /**
     * @brief Constructor with raw stencil information
     * @details This constructor initializes a DiscreteDerivative object with
     * the provided stencil size, relative starting point of stencil, and
     * stencil coefficients.
     * @param nb_pts The stencil size.
     * @param lbounds The relative starting point of stencil. For example, a
     * value of (-2,) means that the stencil starts two pixels to the left of
     * where the derivative should be computed.
     * @param stencil The stencil coefficients.
     */
    DiscreteDerivative(DynCcoord_t nb_pts, DynCcoord_t lbounds,
                       const Eigen::ArrayXd & stencil);

    //! Default copy constructor
    DiscreteDerivative(const DiscreteDerivative & other) = default;

    //! Default move constructor
    DiscreteDerivative(DiscreteDerivative && other) = default;

    //! Default destructor
    virtual ~DiscreteDerivative() = default;

    //! Copy assignment operator is deleted as copying is not allowed for
    //! DiscreteDerivative objects
    DiscreteDerivative & operator=(const DiscreteDerivative & other) = delete;

    //! Move assignment operator is deleted as moving is not allowed for
    //! DiscreteDerivative objects
    DiscreteDerivative & operator=(DiscreteDerivative && other) = delete;

    /**
     * @brief Returns the stencil value at a given coordinate.
     * @details This function returns the stencil value at the provided
     * coordinate.
     * @param dcoord The coordinate at which the stencil value is to be
     * returned.
     * @return The stencil value at the provided coordinate.
     */
    Real operator()(const DynCcoord_t & dcoord) const {
      return this->stencil[this->pixels.get_index(dcoord)];
    }

    /**
     * @brief Returns the dimension of the stencil.
     * @details This function returns the dimension of the stencil, which is the
     * number of spatial dimensions in which the stencil operates. The stencil
     * is a finite-differences stencil used for computing derivatives.
     * @return A constant reference to the Dim_t object that contains the
     * dimension of the stencil.
     */
    const Dim_t & get_dim() const { return this->pixels.get_dim(); }

    /**
     * @brief Returns the number of grid points in the stencil.
     * @details This function returns the number of grid points in the stencil,
     * which is the size of the stencil. The stencil is a finite-differences
     * stencil used for computing derivatives.
     * @return A constant reference to the DynamicCoordinate object that
     * contains the number of grid points in the stencil.
     */
    const DynCcoord_t & get_nb_pts() const {
      return this->pixels.get_nb_subdomain_grid_pts();
    }

    /**
     * @brief Returns the lower bounds of the stencil.
     * @details This function returns the lower bounds of the stencil, which
     * represent the relative starting point of the stencil. For example, a
     * value of (-2,) means that the stencil starts two pixels to the left of
     * where the derivative should be computed.
     * @return A constant reference to the DynamicCoordinate object that
     * contains the lower bounds of the stencil.
     */
    const DynCcoord_t & get_lbounds() const {
      return this->pixels.get_subdomain_locations();
    }

    /**
     * @brief Returns the pixels class that allows to iterate over pixels.
     * @details This function returns the DynamicPixels object from the
     * muGrid::CcoordOps namespace. This object is used to iterate over the
     * pixels of the stencil.
     * @return A constant reference to the DynamicPixels object that allows to
     * iterate over the pixels of the stencil.
     */
    const muGrid::CcoordOps::DynamicPixels & get_pixels() const {
      return this->pixels;
    }

    /**
     * @brief Apply the "stencil" to a component (degree-of-freedom) of a field
     * and store the result to a select component of a second field.
     * @details This function applies the stencil to a component of an input
     * field and stores the result in a selected component of an output field.
     * It performs various checks to ensure the fields are global and the
     * specified degrees of freedom are within range. It then loops over the
     * field pixel iterator and the stencil to compute the derivative and store
     * it in the output field. Note that this function is designed to be inlined
     * by the compiler to optimize loops over degrees of freedom.
     * @note This function currently only works without MPI parallelization. If
     * parallelization is needed, apply the stencil in Fourier space using the
     * `fourier` method. Currently, this method is only used in the serial
     * tests.
     * @tparam T The type of the field elements.
     * @param in_field The input field to which the stencil is applied.
     * @param in_dof The degree of freedom in the input field to which the
     * stencil is applied.
     * @param out_field The output field where the result is stored.
     * @param out_dof The degree of freedom in the output field where the result
     * is stored.
     * @param fac A factor that is multiplied with the derivative. The default
     * value is 1.0.
     * @throws DerivativeError If the input or output field is not global, or if
     * the specified degree of freedom is out of range, or if the input and
     * output fields live on incompatible grids.
     */
    template <typename T>
    void apply(const muGrid::TypedFieldBase<T> & in_field, Index_t in_dof,
               muGrid::TypedFieldBase<T> & out_field, Index_t out_dof,
               Real fac = 1.0) const {
      // check whether fields are global
      if (!in_field.is_global()) {
        throw DerivativeError("Input field must be a global field.");
      }
      if (!out_field.is_global()) {
        throw DerivativeError("Output field must be a global field.");
      }
      // check whether specified dofs are in range
      if (in_dof < 0 or in_dof >= in_field.get_nb_dof_per_pixel()) {
        std::stringstream ss{};
        ss << "Component " << in_dof << " of input field does not exist."
           << "(Input field has " << in_field.get_nb_dof_per_pixel()
           << " components.)";
        throw DerivativeError(ss.str());
      }
      if (out_dof < 0 or out_dof >= out_field.get_nb_dof_per_pixel()) {
        std::stringstream ss{};
        ss << "Component " << out_dof << " of output field does not exist."
           << "(Input field has " << out_field.get_nb_dof_per_pixel()
           << " components.)";
        throw DerivativeError(ss.str());
      }
      // get global field collections
      const auto & in_collection{
          dynamic_cast<const muGrid::GlobalFieldCollection &>(
              in_field.get_collection())};
      const auto & out_collection{
          dynamic_cast<const muGrid::GlobalFieldCollection &>(
              in_field.get_collection())};
      if (in_collection.get_nb_pixels() != out_collection.get_nb_pixels()) {
        std::stringstream ss{};
        ss << "Input fields lives on a " << in_collection.get_nb_pixels()
           << " grid, but output fields lives on an incompatible "
           << out_collection.get_nb_pixels() << " grid.";
        throw DerivativeError(ss.str());
      }

      // construct maps
      muGrid::FieldMap<Real, Mapping::Const> in_map{in_field,
                                                    muGrid::IterUnit::Pixel};
      muGrid::FieldMap<Real, Mapping::Mut> out_map{out_field,
                                                   muGrid::IterUnit::Pixel};
      // loop over field pixel iterator
      Index_t ndim{in_collection.get_spatial_dim()};
      auto & nb_grid_pts{
          in_collection.get_pixels().get_nb_subdomain_grid_pts()};
      in_collection.get_pixels().get_nb_subdomain_grid_pts();
      for (const auto && coord : in_collection.get_pixels()) {
        T derivative{};
        // loop over stencil
        for (const auto && dcoord : this->pixels) {
          auto coord2{coord + dcoord};
          // TODO(pastewka): This only works in serial. For this to work
          //  properly in (MPI) parallel, we need ghost buffers (which will
          //  affect large parts of the code).
          for (Index_t dim{0}; dim < ndim; ++dim) {
            coord2[dim] =
                muGrid::CcoordOps::modulo(coord2[dim], nb_grid_pts[dim]);
          }
          derivative += this->stencil[this->pixels.get_index(dcoord)] *
                        in_map[in_collection.get_index(coord2)](in_dof);
        }
        out_map[out_collection.get_index(coord)](out_dof) = fac * derivative;
      }
    }

    /**
     * @brief Returns the Fourier representation of the stencil.
     * @details Any translationally invariant linear combination of grid values
     * (as expressed through the "stencil") becomes a multiplication with a
     * number in Fourier space. This method returns the Fourier representation
     * of this stencil.
     * @param phase The phase is the wavevector times cell dimension, but
     * lacking a factor of 2 π.
     * @return The Fourier representation of the stencil.
     */
    virtual Complex fourier(const Vector & phase) const {
      Complex s{0, 0};
      for (auto && dcoord : muGrid::CcoordOps::DynamicPixels(
               this->pixels.get_nb_subdomain_grid_pts(),
               this->pixels.get_subdomain_locations())) {
        const Real arg{phase.matrix().dot(eigen(dcoord).template cast<Real>())};
        s += this->operator()(dcoord) *
             std::exp(Complex(0, 2 * muGrid::pi * arg));
      }
      return s;
    }

    /**
     * @brief Returns a new stencil with rolled axes.
     * @details Given a stencil on a three-dimensional grid with axes (x, y, z),
     * the stencil that has been "rolled" by distance one has axes (z, x, y).
     * This is a simple implementation of a rotation operation. For example,
     * given a stencil that described the derivative in the x-direction,
     * rollaxes(1) gives the derivative in the y-direction and rollaxes(2) gives
     * the derivative in the z-direction.
     * @param distance The distance to roll the axes. Default value is 1.
     * @return A new DiscreteDerivative object with rolled axes.
     */
    DiscreteDerivative rollaxes(int distance = 1) const;

    /**
     * @brief Returns the stencil data.
     * @details This function returns the finite-differences stencil data which
     * is used for computing derivatives.
     * @return A constant reference to the Eigen::ArrayXd object that contains
     * the stencil data.
     */
    const Eigen::ArrayXd & get_stencil() const { return this->stencil; }

   protected:
    muGrid::CcoordOps::DynamicPixels pixels{};  //!< iterate over the stencil
    const Eigen::ArrayXd stencil;               //!< Finite-differences stencil
  };

  /**
   * Allows inserting `muFFT::DiscreteDerivative`s into `std::ostream`s
   */
  std::ostream & operator<<(std::ostream & os,
                            const DiscreteDerivative & derivative);

  //! convenience alias
  using Gradient_t = std::vector<std::shared_ptr<DerivativeBase>>;

  /**
   * convenience function to build a spatial_dimension-al gradient operator
   * using exact Fourier differentiation
   *
   * @param spatial_dimension number of spatial dimensions
   */
  Gradient_t make_fourier_gradient(const Index_t & spatial_dimension);
}  // namespace muFFT

#endif  // SRC_LIBMUFFT_DERIVATIVE_HH_
