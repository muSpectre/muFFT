/**
 * @file   fft_engine_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Dec 2017
 *
 * @brief  Interface for FFT engines
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

#ifndef SRC_LIBMUFFT_FFT_ENGINE_BASE_HH_
#define SRC_LIBMUFFT_FFT_ENGINE_BASE_HH_

#include "libmugrid/ccoord_operations.hh"
#include "libmugrid/field_collection_global.hh"
#include "libmugrid/field_typed.hh"

#include "libmugrid/communicator.hh"
#include "mufft_common.hh"

#include <set>

namespace muFFT {
  using muGrid::Communicator;

  /**
   * base class for FFTEngine-related exceptions
   */
  class FFTEngineError : public RuntimeError {
   public:
    //! constructor
    explicit FFTEngineError(const std::string & what) : RuntimeError(what) {}
    //! constructor
    explicit FFTEngineError(const char * what) : RuntimeError(what) {}
  };

  /**
   * Virtual base class for FFT engines. To be implemented by all
   * FFT_engine implementations.
   */
  class FFTEngineBase {
   public:
    //! global FieldCollection
    using GFieldCollection_t = muGrid::GlobalFieldCollection;
    //! pixel iterator
    using Pixels_t = typename GFieldCollection_t::DynamicPixels;
    /**
     * Field type on which to apply the projection.
     * This is a TypedFieldBase because it need to be able to hold
     * either TypedField or a WrappedField.
     */
    using RealField_t = muGrid::TypedFieldBase<Real>;
    /**
     * Field type holding a Fourier-space representation of a
     * real-valued second-order tensor field
     */
    using FourierField_t = muGrid::TypedFieldBase<Complex>;
    /**
     * iterator over Fourier-space discretisation point
     */
    using iterator = typename GFieldCollection_t::DynamicPixels::iterator;

    /**
     * Sub point map type -- for multiple nodal or quad points
     * - map to hold nb_sub_pts by tag
     */
    using SubPtMap_t = std::map<std::string, Index_t>;

    //! Default constructor
    FFTEngineBase() = delete;

    /**
     * @brief Constructs an FFTEngineBase object with the specified parameters.
     *
     * This constructor initializes an FFTEngineBase object with the given
     * number of grid points, communicator, FFT planner flags, and buffer
     * options. The constructor does not perform any FFT computations; it merely
     * sets up the object for future FFT operations.
     *
     * @param nb_grid_pts A DynCcoord_t object representing the number of grid
     * points in each dimension of the global grid.
     * @param comm An optional Communicator object for MPI communication.
     * Defaults to an empty Communicator.
     * @param plan_flags An optional FFT_PlanFlags object representing the FFT
     * planner flags. Defaults to FFT_PlanFlags::estimate.
     * @param allow_temporary_buffer An optional boolean flag indicating whether
     * the creation of temporary buffers is allowed if the input buffer has the
     * wrong memory layout. Defaults to true.
     * @param allow_destroy_input An optional boolean flag indicating whether
     * the input buffers can be invalidated during the FFT. Defaults to false.
     * @param engine_has_rigid_memory_layout An optional boolean flag indicating
     * whether the underlying FFT engine requires a fixed memory layout.
     * Defaults to true.
     */
    FFTEngineBase(const DynCcoord_t & nb_grid_pts,
                  Communicator comm = Communicator(),
                  const FFT_PlanFlags & plan_flags = FFT_PlanFlags::estimate,
                  bool allow_temporary_buffer = true,
                  bool allow_destroy_input = false,
                  bool engine_has_rigid_memory_layout = true);

    //! Copy constructor
    FFTEngineBase(const FFTEngineBase & other) = delete;

    //! Move constructor
    FFTEngineBase(FFTEngineBase && other) = delete;

    //! Destructor
    virtual ~FFTEngineBase() = default;

    //! Copy assignment operator
    FFTEngineBase & operator=(const FFTEngineBase & other) = delete;

    //! Move assignment operator
    FFTEngineBase & operator=(FFTEngineBase && other) = delete;

    /**
     * prepare a plan for a transform with nb_dof_per_pixel entries per pixel.
     * Needs to be called for every different sized transform
     */
    virtual void create_plan(const Index_t & nb_dof_per_pixel) = 0;

    /**
     * prepare a plan for a transform with shape entries per pixel.
     * Needs to be called for every different sized transform
     */
    void create_plan(const Shape_t & shape = Shape_t{});

    //! forward transform, performs copy of buffer if required
    void fft(const RealField_t & input_field, FourierField_t & output_field);

    //! inverse transform, performs copy of buffer if required
    void ifft(const FourierField_t & input_field, RealField_t & output_field);

    //! forward transform using half-complex data storage,
    // performs copy of buffer if required
    void hcfft(const RealField_t & input_field, RealField_t & output_field);

    //! inverse transform using half-complex data storage,
    // performs copy of buffer if required
    void ihcfft(const RealField_t & input_field, RealField_t & output_field);

    /**
     * Create a Fourier-space field with the ideal strides and dimensions for
     * this engine. Fields created this way are meant to be reused again and
     * again, and they will stay in the memory of the `muFFT::FFTEngineBase`'s
     * field collection until the engine is destroyed.
     */
    virtual muGrid::ComplexField &
    register_fourier_space_field(const std::string & unique_name,
                                 const Index_t & nb_dof_per_pixel);

    /**
     * Create a Fourier-space field with the ideal strides and dimensions for
     * this engine. Fields created this way are meant to be reused again and
     * again, and they will stay in the memory of the `muFFT::FFTEngineBase`'s
     * field collection until the engine is destroyed.
     */
    virtual muGrid::ComplexField &
    register_fourier_space_field(const std::string & unique_name,
                                 const Shape_t & shape = Shape_t{});

    /**
     * Fetches a Fourier-space field with the ideal strides and dimensions for
     * this engine. If the field does not exist, it is created using
     * `register_fourier_space_field`.
     */
    FourierField_t & fourier_space_field(const std::string & unique_name,
                                         const Index_t & nb_dof_per_pixel);

    /**
     * Fetches a Fourier-space field with the ideal strides and dimensions for
     * this engine. If the field does not exist, it is created using
     * `register_fourier_space_field`.
     */
    FourierField_t & fourier_space_field(const std::string & unique_name,
                                         const Shape_t & shape = Shape_t{});

    /**
     * Create a Fourier-space field with the ideal strides and dimensions for
     * this engine. Fields created this way are meant to be reused again and
     * again, and they will stay in the memory of the `muFFT::FFTEngineBase`'s
     * field collection until the engine is destroyed.
     */
    virtual RealField_t &
    register_halfcomplex_field(const std::string & unique_name,
                               const Index_t & nb_dof_per_pixel);

    /**
     * Create a Fourier-space field with the ideal strides and dimensions for
     * this engine. Fields created this way are meant to be reused again and
     * again, and they will stay in the memory of the `muFFT::FFTEngineBase`'s
     * field collection until the engine is destroyed.
     */
    virtual RealField_t &
    register_halfcomplex_field(const std::string & unique_name,
                               const Shape_t & shape = Shape_t{});

    /**
     * Fetches a Fourier-space field with the ideal strides and dimensions for
     * this engine. If the field does not exist, it is created using
     * `register_fourier_space_field`.
     */
    RealField_t & halfcomplex_field(const std::string & unique_name,
                                    const Index_t & nb_dof_per_pixel);

    /**
     * Fetches a Fourier-space field with the ideal strides and dimensions for
     * this engine. If the field does not exist, it is created using
     * `register_fourier_space_field`.
     */
    RealField_t & halfcomplex_field(const std::string & unique_name,
                                    const Shape_t & shape = Shape_t{});

    /**
     * Create a real-space field with the ideal strides and dimensions for this
     * engine. Fields created this way are meant to be reused again and again,
     * and they will stay in the memory of the `muFFT::FFTEngineBase`'s field
     * collection until the engine is destroyed.
     */
    virtual RealField_t &
    register_real_space_field(const std::string & unique_name,
                              const Index_t & nb_dof_per_pixel);

    /**
     * Create a real-space field with the ideal strides and dimensions for this
     * engine. Fields created this way are meant to be reused again and again,
     * and they will stay in the memory of the `muFFT::FFTEngineBase`'s field
     * collection until the engine is destroyed.
     */
    virtual RealField_t &
    register_real_space_field(const std::string & unique_name,
                              const Shape_t & shape = Shape_t{},
                              const std::string & sub_division = PixelTag);

    /**
     * Fetches a real-space field with the ideal strides and dimensions for this
     * engine. If the field does not exist, it is created using
     * `register_real_space_field`.
     */
    RealField_t & real_space_field(const std::string & unique_name,
                                   const Index_t & nb_dof_per_pixel);

    /**
     * Fetches a real-space field with the ideal strides and dimensions for this
     * engine. If the field does not exist, it is created using
     * `register_real_space_field`.
     */
    RealField_t & real_space_field(const std::string & unique_name,
                                   const Shape_t & shape = Shape_t{},
                                   const std::string & sub_division = PixelTag);

    //! return whether this engine is active
    virtual bool has_grid_pts() const { return true; }

    /**
     * iterators over only those pixels that exist in real space
     */
    const Pixels_t & get_real_pixels() const;

    /**
     * iterators over only those pixels that exist in frequency space
     * (i.e. about half of all pixels, see rfft)
     */
    const Pixels_t & get_fourier_pixels() const;

    //! nb of pixels (mostly for debugging)
    size_t size() const;
    //! nb of pixels in Fourier space
    size_t fourier_size() const;
    //! nb of pixels in the work space (may contain a padding region)
    size_t workspace_size() const;

    //! return the communicator object
    const Communicator & get_communicator() const { return this->comm; }

    /**
     * returns the process-local number of grid points in each direction of the
     * cell
     */
    const DynCcoord_t & get_nb_subdomain_grid_pts() const {
      return this->nb_subdomain_grid_pts;
    }

    /**
     * returns the global number of grid points in each direction of the cell
     */
    const DynCcoord_t & get_nb_domain_grid_pts() const {
      return this->nb_domain_grid_pts;
    }

    //! returns the process-local locations of the cell
    const DynCcoord_t & get_subdomain_locations() const {
      return this->subdomain_locations;
    }

    //! returns the data layout of the process-local grid
    const DynCcoord_t & get_subdomain_strides() const {
      return this->subdomain_strides;
    }

    /**
     * returns the process-local number of grid points in each direction of the
     * cell in Fourier space
     */
    const DynCcoord_t & get_nb_fourier_grid_pts() const {
      return this->nb_fourier_grid_pts;
    }
    //! returns the process-local locations of the cell in Fourier space
    const DynCcoord_t & get_fourier_locations() const {
      return this->fourier_locations;
    }
    //! returns the data layout of the cell in Fourier space
    const DynCcoord_t & get_fourier_strides() const {
      return this->fourier_strides;
    }

    //! returns the field collection handling fields in real space
    GFieldCollection_t & get_real_field_collection() {
      return this->real_field_collection;
    }

    //! returns the field collection handling fields confirming with
    // the data layout required for half-complex transforms
    GFieldCollection_t & get_halfcomplex_field_collection() {
      return this->halfcomplex_field_collection;
    }

    //! returns the field collection handling fields in Fourier space
    GFieldCollection_t & get_fourier_field_collection() {
      return this->fourier_field_collection;
    }

    //! factor by which to multiply projection before inverse transform (this is
    //! typically 1/nb_pixels for so-called unnormalized transforms (see,
    //! e.g.
    //! http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data
    //! or https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.fft.html
    //! . Rather than scaling the inverse transform (which would cost one more
    //! loop), FFT engines provide this value so it can be used in the
    //! projection operator (where no additional loop is required)
    inline Real normalisation() const { return norm_factor; }

    //! return the number of spatial dimensions
    const Index_t & get_spatial_dim() const;

    //! return whether temporary buffers are allowed
    bool get_allow_temporary_buffer() const { return allow_temporary_buffer; }

    //! perform a deep copy of the engine (this should never be necessary in
    //! c++)
    virtual std::unique_ptr<FFTEngineBase> clone() const = 0;

    //! check whether a plan for nb_dof_per_pixel exists
    bool has_plan_for(const Index_t & nb_dof_per_pixel) const;

   protected:
    //! calls initialize of the real, hc and fourier field collections
    void initialise_field_collections();

    //! forward transform, assumes that the buffer has the correct memory layout
    virtual void compute_fft(const RealField_t & input_field,
                             FourierField_t & output_field) = 0;

    //! inverse transform, assumes that the buffer has the correct memory layout
    virtual void compute_ifft(const FourierField_t & input_field,
                              RealField_t & output_field) = 0;

    //! forward half complex transform
    virtual void compute_hcfft(const RealField_t & input_field,
                               RealField_t & output_field);

    //! inverse half complex transform
    virtual void compute_ihcfft(const RealField_t & input_field,
                                RealField_t & output_field);

    //! check whether real-space buffer has the correct memory layout
    virtual bool check_real_space_field(const RealField_t & field,
                                        FFTDirection direction) const;

    //! check whether Fourier-space buffer has the correct memory layout
    virtual bool check_fourier_space_field(const FourierField_t & field,
                                           FFTDirection direction) const;

    //! check whether the half-complex buffer has the correct memory layout
    virtual bool check_halfcomplex_field(const RealField_t & field,
                                         FFTDirection direction) const;

    //! spatial dimension of the grid
    Index_t spatial_dimension;
    /**
     * Field collection in which to store fields associated with
     * Fourier-space points
     */
    Communicator comm;  //!< communicator
    //! Field collection for real-space fields
    GFieldCollection_t real_field_collection;
    //! Field collection for Fourier-space fields
    GFieldCollection_t fourier_field_collection;
    //! Field collection for half-complex-space fields
    //! in the r2hc transform real fields and fourier fields
    //! are identical
    //! In serial the hc_field is identical to the real field,
    //! But in parallel, the hc_field has no padding region.
    GFieldCollection_t halfcomplex_field_collection;

    //! nb_grid_pts of the full domain of the cell
    const DynCcoord_t nb_domain_grid_pts;

    //! nb_grid_pts of the process-local (subdomain) portion of the cell
    DynCcoord_t nb_subdomain_grid_pts;
    //! location of the process-local (subdomain) portion of the cell
    DynCcoord_t subdomain_locations;
    //! data layout of the porcess-local portion of the cell
    DynCcoord_t subdomain_strides;
    //! nb_grid_pts of the process-local (subdomain) portion of the Fourier
    //! transformed data
    DynCcoord_t nb_fourier_grid_pts;
    //! location of the process-local (subdomain) portion of the Fourier
    //! transformed data
    DynCcoord_t fourier_locations;
    //! data layout of the process-local (subdomain) portion of the Fourier
    //! transformed data
    DynCcoord_t fourier_strides;

    //! allow the FFTEngine to create temporary copies (if it cannot work with
    //! a specific memory layout)
    bool allow_temporary_buffer;

    //! allow the FFTEngine to destroy input buffers
    bool allow_destroy_input;

    //! the underlying FFT engine requires a fixed memory layout
    bool engine_has_rigid_memory_layout;

    //! normalisation coefficient of fourier transform
    const Real norm_factor;

    //! FFT planner flags
    const FFT_PlanFlags plan_flags;

    //! number of degrees of freedom per pixel for which this field collection
    //! has been primed. Can be queried. Corresponds to the number of sub-points
    //! per pixel multiplied by the number of components per sub-point
    std::set<Index_t> planned_nb_dofs{};
  };

  //! reference to fft engine is safely managed through a `std::shared_ptr`
  using FFTEngine_ptr = std::shared_ptr<FFTEngineBase>;

}  // namespace muFFT

#endif  // SRC_LIBMUFFT_FFT_ENGINE_BASE_HH_
