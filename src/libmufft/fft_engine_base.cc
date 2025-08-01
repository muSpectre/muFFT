
/**
 * @file   fft_engine_base.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Dec 2017
 *
 * @brief  implementation for FFT engine base class
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

#include "fft_engine_base.hh"
#include "fft_utils.hh"
#include "libmugrid/ccoord_operations.hh"

using muGrid::GlobalFieldCollection;
using muGrid::CcoordOps::get_col_major_strides;
using muGrid::operator<<;

namespace muFFT {

  /* ---------------------------------------------------------------------- */
  FFTEngineBase::FFTEngineBase(const DynCcoord_t & nb_grid_pts,
                               Communicator comm,
                               const FFT_PlanFlags & plan_flags,
                               bool allow_temporary_buffer,
                               bool allow_destroy_input,
                               bool engine_has_rigid_memory_layout)
      : spatial_dimension{nb_grid_pts.get_dim()}, comm{comm},
        real_field_collection{this->spatial_dimension,
                              GlobalFieldCollection::SubPtMap_t{{PixelTag, 1}}},
        fourier_field_collection{
            this->spatial_dimension,
            GlobalFieldCollection::SubPtMap_t{{PixelTag, 1}}},
        halfcomplex_field_collection{
            this->spatial_dimension,
            GlobalFieldCollection::SubPtMap_t{{PixelTag, 1}}},
        nb_domain_grid_pts{nb_grid_pts}, nb_subdomain_grid_pts{nb_grid_pts},
        subdomain_locations(spatial_dimension),
        subdomain_strides{get_col_major_strides(nb_grid_pts)},
        nb_fourier_grid_pts{get_nb_hermitian_grid_pts(nb_grid_pts)},
        fourier_locations(spatial_dimension),
        fourier_strides{get_col_major_strides(nb_fourier_grid_pts)},
        allow_temporary_buffer{allow_temporary_buffer},
        allow_destroy_input{allow_destroy_input},
        engine_has_rigid_memory_layout{engine_has_rigid_memory_layout},
        norm_factor{1. / muGrid::CcoordOps::get_size(nb_grid_pts)},
        plan_flags{plan_flags} {}

  /* ---------------------------------------------------------------------- */
  //! forward transform, performs copy of buffer if required
  void FFTEngineBase::fft(const RealField_t & input_field,
                          FourierField_t & output_field) {
    // Sanity check 1: Do we have a plan?
    auto && nb_dof_per_pixel{input_field.get_nb_dof_per_pixel()};
    if (not this->has_plan_for(nb_dof_per_pixel)) {
      std::stringstream message{};
      message << "No plan has been created for " << nb_dof_per_pixel
              << " degrees of freedom per pixel on MPI rank "
              << this->comm.rank() << ". Use muFFT::FFTEngineBase::create_plan`"
              << " to prepare a plan.";
      throw FFTEngineError{message.str()};
    }

    // Sanity check 2: Does the input field have the correct number of pixels?
    if (static_cast<size_t>(input_field.get_nb_pixels()) !=
        muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)) {
      std::stringstream error{};
      error << "The number of pixels of the field '" << input_field.get_name()
            << "' passed to the forward FFT is " << input_field.get_nb_pixels()
            << " and does not match the size "
            << muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)
            << " of the (sub)domain handled by this FFT engine.";
      throw FFTEngineError(error.str());
    }

    // Sanity check 3: Does the output field have the correct number of pixels?
    if (static_cast<size_t>(output_field.get_nb_pixels()) !=
        muGrid::CcoordOps::get_size(this->nb_fourier_grid_pts)) {
      std::stringstream error{};
      error << "The number of pixels of the field '" << output_field.get_name()
            << "' passed to the forward FFT is " << output_field.get_nb_pixels()
            << " and does not match the size "
            << muGrid::CcoordOps::get_size(this->nb_fourier_grid_pts)
            << " of the (sub)domain handled by this FFT engine.";
      throw FFTEngineError(error.str());
    }

    // Sanity check 4: Do both fields have the same number of DOFs per pixel?
    if (input_field.get_nb_dof_per_pixel() !=
        output_field.get_nb_dof_per_pixel()) {
      std::stringstream error;
      error << "The input field reports " << input_field.get_nb_components()
            << " components per sub-point and " << input_field.get_nb_sub_pts()
            << " sub-points, while the output field reports "
            << output_field.get_nb_components()
            << " components per sub-point and " << output_field.get_nb_sub_pts()
            << " sub-points.";
      throw FFTEngineError(error.str());
    }

    bool input_copy_necessary{
        !this->check_real_space_field(input_field, FFTDirection::forward)};
    bool output_copy_necessary{
        !this->check_fourier_space_field(output_field, FFTDirection::forward)};
    if (this->allow_temporary_buffer and
        (input_copy_necessary or output_copy_necessary)) {
      if (input_copy_necessary and output_copy_necessary) {
        std::stringstream iname, oname;
        iname << "temp_real_space_" << input_field.get_nb_dof_per_pixel();
        oname << "temp_fourier_space_" << output_field.get_nb_dof_per_pixel();
        RealField_t & tmp_ifield{this->real_space_field(
            iname.str(), input_field.get_nb_dof_per_pixel())};
        tmp_ifield.get_collection().set_nb_sub_pts(
            input_field.get_sub_division_tag(), input_field.get_nb_sub_pts());
        tmp_ifield.reshape(input_field.get_components_shape(),
                           input_field.get_sub_division_tag());
        tmp_ifield = input_field;
        FourierField_t & tmp_ofield{this->fourier_space_field(
            oname.str(), output_field.get_nb_dof_per_pixel())};
        tmp_ofield.get_collection().set_nb_sub_pts(
            output_field.get_sub_division_tag(), output_field.get_nb_sub_pts());
        tmp_ofield.reshape(output_field.get_components_shape(),
                           output_field.get_sub_division_tag());
        this->compute_fft(tmp_ifield, tmp_ofield);
        output_field = tmp_ofield;
      } else if (input_copy_necessary) {
        std::stringstream iname;
        iname << "temp_real_space_" << input_field.get_nb_dof_per_pixel();
        RealField_t & tmp_ifield{this->real_space_field(
            iname.str(), input_field.get_nb_dof_per_pixel())};
        tmp_ifield.get_collection().set_nb_sub_pts(
            input_field.get_sub_division_tag(), input_field.get_nb_sub_pts());
        tmp_ifield.reshape(input_field.get_components_shape(),
                           input_field.get_sub_division_tag());
        tmp_ifield = input_field;
        this->compute_fft(tmp_ifield, output_field);
      } else {  // output_copy_necessary
        std::stringstream oname;
        oname << "temp_fourier_space_" << output_field.get_nb_dof_per_pixel();
        FourierField_t & tmp_ofield{this->fourier_space_field(
            oname.str(), output_field.get_nb_dof_per_pixel())};
        tmp_ofield.get_collection().set_nb_sub_pts(
            output_field.get_sub_division_tag(), output_field.get_nb_sub_pts());
        tmp_ofield.reshape(output_field.get_components_shape(),
                           output_field.get_sub_division_tag());
        this->compute_fft(input_field, tmp_ofield);
        output_field = tmp_ofield;
      }
    } else {
      //! no temporary buffers allowwd
      if (input_copy_necessary) {
        throw FFTEngineError("Incompatible memory layout for the real-space "
                             "field and no temporary copies are allowed.");
      }
      if (output_copy_necessary) {
        throw FFTEngineError("Incompatible memory layout for the "
                             "Fourier-space field and no temporary copies are "
                             "allowed.");
      }
      this->compute_fft(input_field, output_field);
    }
  }

  //! inverse transform, performs copy of buffer if required
  void FFTEngineBase::ifft(const FourierField_t & input_field,
                           RealField_t & output_field) {
    // Sanity check 1: Do we have a plan?
    auto && nb_dof_per_pixel{input_field.get_nb_dof_per_pixel()};
    if (not this->has_plan_for(nb_dof_per_pixel)) {
      std::stringstream message{};
      message << "No plan has been created for " << nb_dof_per_pixel
              << " degrees of freedom per pixel on MPI rank "
              << this->comm.rank() << ". Use muFFT::FFTEngineBase::create_plan`"
              << " to prepare a plan.";
      throw FFTEngineError{message.str()};
    }

    // Sanity check 2: Does the input field have the correct number of pixels?
    if (static_cast<size_t>(input_field.get_nb_pixels()) !=
        muGrid::CcoordOps::get_size(this->nb_fourier_grid_pts)) {
      std::stringstream error;
      error << "The number of pixels of the field '" << input_field.get_name()
            << "' passed to the inverse FFT is " << input_field.get_nb_pixels()
            << " and does not match the size "
            << muGrid::CcoordOps::get_size(this->nb_fourier_grid_pts)
            << " of the (sub)domain handled by this FFT engine.";
      throw FFTEngineError(error.str());
    }

    // Sanity check 3: Does the output field have the correct number of pixels?
    if (static_cast<size_t>(output_field.get_nb_pixels()) !=
        muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)) {
      std::stringstream error;
      error << "The number of pixels of the field '" << output_field.get_name()
            << "' passed to the inverse FFT is " << output_field.get_nb_pixels()
            << " and does not match the size "
            << muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)
            << " of the (sub)domain handled by this FFT engine.";
      throw FFTEngineError(error.str());
    }

    // Sanity check 4: Do both fields have the same number of DOFs per pixel?
    if (input_field.get_nb_dof_per_pixel() !=
        output_field.get_nb_dof_per_pixel()) {
      std::stringstream error;
      error << "The input field reports " << input_field.get_nb_components()
            << " components per sub-point and " << input_field.get_nb_sub_pts()
            << " sub-points, while the output field reports "
            << output_field.get_nb_components()
            << " components per sub-point and " << output_field.get_nb_sub_pts()
            << " sub-points.";
      throw FFTEngineError(error.str());
    }

    bool input_copy_necessary{not this->check_fourier_space_field(
        input_field, FFTDirection::reverse)};
    bool output_copy_necessary{
        not this->check_real_space_field(output_field, FFTDirection::reverse)};
    if (this->allow_temporary_buffer and
        (input_copy_necessary or output_copy_necessary)) {
      if (input_copy_necessary and output_copy_necessary) {
        std::stringstream iname, oname;
        iname << "temp_fourier_space_" << input_field.get_nb_dof_per_pixel();
        oname << "temp_real_space_" << output_field.get_nb_dof_per_pixel();
        FourierField_t & tmp_ifield{this->fourier_space_field(
            iname.str(), input_field.get_nb_dof_per_pixel())};
        tmp_ifield.get_collection().set_nb_sub_pts(
            input_field.get_sub_division_tag(), input_field.get_nb_sub_pts());
        tmp_ifield.reshape(input_field.get_components_shape(),
                           input_field.get_sub_division_tag());
        tmp_ifield = input_field;
        RealField_t & tmp_ofield{this->real_space_field(
            oname.str(), output_field.get_nb_dof_per_pixel())};
        tmp_ofield.get_collection().set_nb_sub_pts(
            output_field.get_sub_division_tag(), output_field.get_nb_sub_pts());
        tmp_ofield.reshape(output_field.get_components_shape(),
                           output_field.get_sub_division_tag());
        this->compute_ifft(tmp_ifield, tmp_ofield);
        output_field = tmp_ofield;
      } else if (input_copy_necessary) {
        std::stringstream iname;
        iname << "temp_fourier_space_" << input_field.get_nb_dof_per_pixel();
        FourierField_t & tmp_ifield{this->fourier_space_field(
            iname.str(), input_field.get_nb_dof_per_pixel())};
        tmp_ifield.get_collection().set_nb_sub_pts(
            input_field.get_sub_division_tag(), input_field.get_nb_sub_pts());
        tmp_ifield.reshape(input_field.get_components_shape(),
                           input_field.get_sub_division_tag());
        tmp_ifield = input_field;
        this->compute_ifft(tmp_ifield, output_field);
      } else {  // output_copy_necessary
        std::stringstream oname;
        oname << "temp_real_space_" << output_field.get_nb_dof_per_pixel();
        RealField_t & tmp_ofield{this->real_space_field(
            oname.str(), output_field.get_nb_dof_per_pixel())};
        tmp_ofield.get_collection().set_nb_sub_pts(
            output_field.get_sub_division_tag(), output_field.get_nb_sub_pts());
        tmp_ofield.reshape(output_field.get_components_shape(),
                           output_field.get_sub_division_tag());
        this->compute_ifft(input_field, tmp_ofield);
        output_field = tmp_ofield;
      }
    } else {
      //! no temporary buffers allowwd
      if (input_copy_necessary) {
        throw FFTEngineError("Incompatible memory layout for the "
                             "Fourier-space field and no temporary copies are "
                             "allowed.");
      }
      if (output_copy_necessary) {
        throw FFTEngineError("Incompatible memory layout for the real-space "
                             "field and no temporary copies are allowed.");
      }
      this->compute_ifft(input_field, output_field);
    }
  }

  //! forward transform, performs copy of buffer if required
  void FFTEngineBase::hcfft(const RealField_t & input_field,
                            RealField_t & output_field) {
    // Sanity check 1: Do we have a plan?
    auto && nb_dof_per_pixel{input_field.get_nb_dof_per_pixel()};
    if (not this->has_plan_for(nb_dof_per_pixel)) {
      std::stringstream message{};
      message << "No plan has been created for " << nb_dof_per_pixel
              << " degrees of freedom per pixel on MPI rank "
              << this->comm.rank() << ". Use muFFT::FFTEngineBase::create_plan`"
              << " to prepare a plan.";
      throw FFTEngineError{message.str()};
    }

    // Sanity check 2: Does the input field have the correct number of pixels?
    if (static_cast<size_t>(input_field.get_nb_pixels()) !=
        muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)) {
      std::stringstream error{};
      error << "The number of pixels of the field '" << input_field.get_name()
            << "' passed to the forward FFT is " << input_field.get_nb_pixels()
            << " and does not match the size "
            << muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)
            << " of the (sub)domain handled by this FFT engine.";
      throw FFTEngineError(error.str());
    }

    // Sanity check 3: Does the output field have the correct number of pixels?
    if (static_cast<size_t>(output_field.get_nb_pixels()) !=
        muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)) {
      std::stringstream error{};
      error << "The number of pixels of the field '" << output_field.get_name()
            << "' passed to the forward FFT is " << output_field.get_nb_pixels()
            << " and does not match the size "
            << muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)
            << " of the (sub)domain handled by this FFT engine.";
      throw FFTEngineError(error.str());
    }

    // Sanity check 4: Do both fields have the same number of DOFs per pixel?
    if (input_field.get_nb_dof_per_pixel() !=
        output_field.get_nb_dof_per_pixel()) {
      std::stringstream error;
      error << "The input field reports " << input_field.get_nb_components()
            << " components per sub-point and " << input_field.get_nb_sub_pts()
            << " sub-points, while the output field reports "
            << output_field.get_nb_components()
            << " components per sub-point and " << output_field.get_nb_sub_pts()
            << " sub-points.";
      throw FFTEngineError(error.str());
    }

    bool input_copy_necessary{
        !this->check_halfcomplex_field(input_field, FFTDirection::forward)};
    bool output_copy_necessary{
        !this->check_halfcomplex_field(output_field, FFTDirection::forward)};
    if (this->allow_temporary_buffer and
        (input_copy_necessary or output_copy_necessary)) {
      if (input_copy_necessary and output_copy_necessary) {
        std::stringstream iname, oname;
        iname << "temp_real_space_" << input_field.get_nb_dof_per_pixel();
        oname << "temp_fourier_space_" << output_field.get_nb_dof_per_pixel();
        RealField_t & tmp_ifield{this->real_space_field(
            iname.str(), input_field.get_nb_dof_per_pixel())};
        tmp_ifield.get_collection().set_nb_sub_pts(
            input_field.get_sub_division_tag(), input_field.get_nb_sub_pts());
        tmp_ifield.reshape(input_field.get_components_shape(),
                           input_field.get_sub_division_tag());
        tmp_ifield = input_field;
        RealField_t & tmp_ofield{this->halfcomplex_field(
            oname.str(), output_field.get_nb_dof_per_pixel())};
        tmp_ofield.get_collection().set_nb_sub_pts(
            output_field.get_sub_division_tag(), output_field.get_nb_sub_pts());
        tmp_ofield.reshape(output_field.get_components_shape(),
                           output_field.get_sub_division_tag());
        this->compute_hcfft(tmp_ifield, tmp_ofield);
        output_field = tmp_ofield;
      } else if (input_copy_necessary) {
        std::stringstream iname;
        iname << "temp_real_space_" << input_field.get_nb_dof_per_pixel();
        RealField_t & tmp_ifield{this->real_space_field(
            iname.str(), input_field.get_nb_dof_per_pixel())};
        tmp_ifield.get_collection().set_nb_sub_pts(
            input_field.get_sub_division_tag(), input_field.get_nb_sub_pts());
        tmp_ifield.reshape(input_field.get_components_shape(),
                           input_field.get_sub_division_tag());
        tmp_ifield = input_field;
        this->compute_hcfft(tmp_ifield, output_field);
      } else {  // output_copy_necessary
        std::stringstream oname;
        oname << "temp_fourier_space_" << output_field.get_nb_dof_per_pixel();
        RealField_t & tmp_ofield{this->halfcomplex_field(
            oname.str(), output_field.get_nb_dof_per_pixel())};
        tmp_ofield.get_collection().set_nb_sub_pts(
            output_field.get_sub_division_tag(), output_field.get_nb_sub_pts());
        tmp_ofield.reshape(output_field.get_components_shape(),
                           output_field.get_sub_division_tag());
        this->compute_hcfft(input_field, tmp_ofield);
        output_field = tmp_ofield;
      }
    } else {
      //! no temporary buffers allowwd
      if (input_copy_necessary) {
        throw FFTEngineError("Incompatible memory layout for the real-space "
                             "field and no temporary copies are allowed.");
      }
      if (output_copy_necessary) {
        throw FFTEngineError("Incompatible memory layout for the "
                             "Fourier-space field and no temporary copies are "
                             "allowed.");
      }
      this->compute_hcfft(input_field, output_field);
    }
  }

  //! inverse transform, performs copy of buffer if required
  void FFTEngineBase::ihcfft(const RealField_t & input_field,
                             RealField_t & output_field) {
    // Sanity check 1: Do we have a plan?
    auto && nb_dof_per_pixel{input_field.get_nb_dof_per_pixel()};
    if (not this->has_plan_for(nb_dof_per_pixel)) {
      std::stringstream message{};
      message << "No plan has been created for " << nb_dof_per_pixel
              << " degrees of freedom per pixel on MPI rank "
              << this->comm.rank() << ". Use muFFT::FFTEngineBase::create_plan`"
              << " to prepare a plan.";
      throw FFTEngineError{message.str()};
    }

    // Sanity check 2: Does the input field have the correct number of pixels?
    if (static_cast<size_t>(input_field.get_nb_pixels()) !=
        muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)) {
      std::stringstream error;
      error << "The number of pixels of the field '" << input_field.get_name()
            << "' passed to the inverse FFT is " << input_field.get_nb_pixels()
            << " and does not match the size "
            << muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)
            << " of the (sub)domain handled by this FFT engine.";
      throw FFTEngineError(error.str());
    }

    // Sanity check 3: Does the output field have the correct number of pixels?
    if (static_cast<size_t>(output_field.get_nb_pixels()) !=
        muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)) {
      std::stringstream error;
      error << "The number of pixels of the field '" << output_field.get_name()
            << "' passed to the inverse FFT is " << output_field.get_nb_pixels()
            << " and does not match the size "
            << muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)
            << " of the (sub)domain handled by this FFT engine.";
      throw FFTEngineError(error.str());
    }

    // Sanity check 4: Do both fields have the same number of DOFs per pixel?
    if (input_field.get_nb_dof_per_pixel() !=
        output_field.get_nb_dof_per_pixel()) {
      std::stringstream error;
      error << "The input field reports " << input_field.get_nb_components()
            << " components per sub-point and " << input_field.get_nb_sub_pts()
            << " sub-points, while the output field reports "
            << output_field.get_nb_components()
            << " components per sub-point and " << output_field.get_nb_sub_pts()
            << " sub-points.";
      throw FFTEngineError(error.str());
    }

    bool input_copy_necessary{
        not this->check_halfcomplex_field(input_field, FFTDirection::reverse)};
    bool output_copy_necessary{
        not this->check_real_space_field(output_field, FFTDirection::reverse)};
    if (this->allow_temporary_buffer and
        (input_copy_necessary or output_copy_necessary)) {
      if (input_copy_necessary and output_copy_necessary) {
        std::stringstream iname, oname;
        iname << "temp_fourier_space_" << input_field.get_nb_dof_per_pixel();
        oname << "temp_real_space_" << output_field.get_nb_dof_per_pixel();
        RealField_t & tmp_ifield{this->halfcomplex_field(
            iname.str(), input_field.get_nb_dof_per_pixel())};
        tmp_ifield.get_collection().set_nb_sub_pts(
            input_field.get_sub_division_tag(), input_field.get_nb_sub_pts());
        tmp_ifield.reshape(input_field.get_components_shape(),
                           input_field.get_sub_division_tag());
        tmp_ifield = input_field;
        RealField_t & tmp_ofield{this->real_space_field(
            oname.str(), output_field.get_nb_dof_per_pixel())};
        tmp_ofield.get_collection().set_nb_sub_pts(
            output_field.get_sub_division_tag(), output_field.get_nb_sub_pts());
        tmp_ofield.reshape(output_field.get_components_shape(),
                           output_field.get_sub_division_tag());
        this->compute_ihcfft(tmp_ifield, tmp_ofield);
        output_field = tmp_ofield;
      } else if (input_copy_necessary) {
        std::stringstream iname;
        iname << "temp_fourier_space_" << input_field.get_nb_dof_per_pixel();
        RealField_t & tmp_ifield{this->halfcomplex_field(
            iname.str(), input_field.get_nb_dof_per_pixel())};
        tmp_ifield.get_collection().set_nb_sub_pts(
            input_field.get_sub_division_tag(), input_field.get_nb_sub_pts());
        tmp_ifield.reshape(input_field.get_components_shape(),
                           input_field.get_sub_division_tag());
        tmp_ifield = input_field;
        this->compute_ihcfft(tmp_ifield, output_field);
      } else {  // output_copy_necessary
        std::stringstream oname;
        oname << "temp_real_space_" << output_field.get_nb_dof_per_pixel();
        RealField_t & tmp_ofield{this->real_space_field(
            oname.str(), output_field.get_nb_dof_per_pixel())};
        tmp_ofield.get_collection().set_nb_sub_pts(
            output_field.get_sub_division_tag(), output_field.get_nb_sub_pts());
        tmp_ofield.reshape(output_field.get_components_shape(),
                           output_field.get_sub_division_tag());
        this->compute_ihcfft(input_field, tmp_ofield);
        output_field = tmp_ofield;
      }
    } else {
      //! no temporary buffers allowwd
      if (input_copy_necessary) {
        throw FFTEngineError("Incompatible memory layout for the "
                             "Fourier-space field and no temporary copies are "
                             "allowed.");
      }
      if (output_copy_necessary) {
        throw FFTEngineError("Incompatible memory layout for the real-space "
                             "field and no temporary copies are allowed.");
      }
      this->compute_ihcfft(input_field, output_field);
    }
  }

  /* ---------------------------------------------------------------------- */
  auto
  FFTEngineBase::register_fourier_space_field(const std::string & unique_name,
                                              const Index_t & nb_dof_per_pixel)
      -> muGrid::ComplexField & {
    this->create_plan(nb_dof_per_pixel);
    return this->fourier_field_collection.register_complex_field(
        unique_name, nb_dof_per_pixel, PixelTag);
  }

  /* ---------------------------------------------------------------------- */
  auto
  FFTEngineBase::register_fourier_space_field(const std::string & unique_name,
                                              const Shape_t & shape)
      -> muGrid::ComplexField & {
    this->create_plan(shape);
    return this->fourier_field_collection.register_complex_field(
        unique_name, shape, PixelTag);
  }

  /* ---------------------------------------------------------------------- */
  auto FFTEngineBase::fourier_space_field(const std::string & unique_name,
                                          const Index_t & nb_dof_per_pixel)
      -> FourierField_t & {
    this->create_plan(nb_dof_per_pixel);
    if (this->fourier_field_collection.field_exists(unique_name)) {
      auto & field{dynamic_cast<FourierField_t &>(
          this->fourier_field_collection.get_field(unique_name))};
      if (field.get_nb_dof_per_pixel() != nb_dof_per_pixel) {
        std::stringstream message{};
        message << "Field '" << unique_name << "' exists, but it has "
                << field.get_nb_dof_per_pixel()
                << " degrees of freedom per pixel instead of the requested "
                << nb_dof_per_pixel << ".";
        throw FFTEngineError{message.str()};
      }
      return field;
    }
    return this->register_fourier_space_field(unique_name, nb_dof_per_pixel);
  }

  /* ---------------------------------------------------------------------- */
  auto FFTEngineBase::fourier_space_field(const std::string & unique_name,
                                          const Shape_t & shape)
      -> FourierField_t & {
    this->create_plan(shape);
    if (this->fourier_field_collection.field_exists(unique_name)) {
      auto & field{dynamic_cast<FourierField_t &>(
          this->fourier_field_collection.get_field(unique_name))};
      if (field.get_components_shape() != shape) {
        std::stringstream message{};
        message << "Field '" << unique_name << "' exists, but it has shape of "
                << field.get_components_shape() << " instead of the requested "
                << shape << ".";
        throw FFTEngineError{message.str()};
      }
      return field;
    }
    return this->register_fourier_space_field(unique_name, shape);
  }

  auto
  FFTEngineBase::register_halfcomplex_field(const std::string & unique_name,
                                            const Index_t & nb_dof_per_pixel)
      -> RealField_t & {
    this->create_plan(nb_dof_per_pixel);
    return this->halfcomplex_field_collection.register_real_field(
        unique_name, nb_dof_per_pixel, PixelTag);
  }

  /* ---------------------------------------------------------------------- */
  auto FFTEngineBase::register_halfcomplex_field(
      const std::string & unique_name, const Shape_t & shape) -> RealField_t & {
    this->create_plan(shape);
    return this->halfcomplex_field_collection.register_real_field(
        unique_name, shape, PixelTag);
  }

  /* ---------------------------------------------------------------------- */
  auto FFTEngineBase::halfcomplex_field(const std::string & unique_name,
                                        const Index_t & nb_dof_per_pixel)
      -> RealField_t & {
    this->create_plan(nb_dof_per_pixel);
    if (this->halfcomplex_field_collection.field_exists(unique_name)) {
      auto & field{dynamic_cast<RealField_t &>(
          this->halfcomplex_field_collection.get_field(unique_name))};
      if (field.get_nb_dof_per_pixel() != nb_dof_per_pixel) {
        std::stringstream message{};
        message << "Field '" << unique_name << "' exists, but it has "
                << field.get_nb_dof_per_pixel()
                << " degrees of freedom per pixel instead of the requested "
                << nb_dof_per_pixel << ".";
        throw FFTEngineError{message.str()};
      }
      return field;
    }
    return this->register_halfcomplex_field(unique_name, nb_dof_per_pixel);
  }

  /* ---------------------------------------------------------------------- */
  auto FFTEngineBase::halfcomplex_field(const std::string & unique_name,
                                        const Shape_t & shape)
      -> RealField_t & {
    this->create_plan(shape);
    if (this->halfcomplex_field_collection.field_exists(unique_name)) {
      auto & field{dynamic_cast<RealField_t &>(
          this->halfcomplex_field_collection.get_field(unique_name))};
      if (field.get_components_shape() != shape) {
        std::stringstream message{};
        message << "Field '" << unique_name << "' exists, but it has shape of "
                << field.get_components_shape() << " instead of the requested "
                << shape << ".";
        throw FFTEngineError{message.str()};
      }
      return field;
    }
    return this->register_halfcomplex_field(unique_name, shape);
  }

  /* ---------------------------------------------------------------------- */

  /* ---------------------------------------------------------------------- */
  auto
  FFTEngineBase::register_real_space_field(const std::string & unique_name,
                                           const Index_t & nb_dof_per_pixel)
      -> RealField_t & {
    this->create_plan(nb_dof_per_pixel);
    return this->real_field_collection.register_real_field(
        unique_name, nb_dof_per_pixel, PixelTag);
  }

  /* ---------------------------------------------------------------------- */
  auto FFTEngineBase::register_real_space_field(
      const std::string & unique_name, const Shape_t & shape,
      const std::string & sub_division) -> RealField_t & {
    this->create_plan(shape);
    return this->real_field_collection.register_real_field(unique_name, shape,
                                                           sub_division);
  }

  /* ---------------------------------------------------------------------- */
  auto FFTEngineBase::real_space_field(const std::string & unique_name,
                                       const Index_t & nb_dof_per_pixel)
      -> RealField_t & {
    this->create_plan(nb_dof_per_pixel);
    if (this->real_field_collection.field_exists(unique_name)) {
      auto & field{dynamic_cast<RealField_t &>(
          this->real_field_collection.get_field(unique_name))};
      if (field.get_nb_dof_per_pixel() != nb_dof_per_pixel) {
        std::stringstream message{};
        message << "Field '" << unique_name << "' exists, but it has "
                << field.get_nb_dof_per_pixel()
                << " degrees of freedom per pixel instead of the requested "
                << nb_dof_per_pixel << ".";
        throw muGrid::FieldCollectionError{message.str()};
      }
      return field;
    }
    return this->register_real_space_field(unique_name, nb_dof_per_pixel);
  }

  /* ---------------------------------------------------------------------- */
  auto FFTEngineBase::real_space_field(const std::string & unique_name,
                                       const Shape_t & shape,
                                       const std::string & sub_divison)
      -> RealField_t & {
    this->create_plan(shape);
    if (this->real_field_collection.field_exists(unique_name)) {
      auto & field{dynamic_cast<RealField_t &>(
          this->real_field_collection.get_field(unique_name))};
      if (field.get_components_shape() != shape) {
        std::stringstream message{};
        message << "Field '" << unique_name << "' exists, but it has shape of "
                << field.get_components_shape() << " instead of the requested "
                << shape << ".";
        throw muGrid::FieldCollectionError{message.str()};
      }
      return field;
    }
    return this->register_real_space_field(unique_name, shape, sub_divison);
  }

  /* ---------------------------------------------------------------------- */
  void FFTEngineBase::create_plan(const Shape_t & shape) {
    this->create_plan(std::accumulate(shape.begin(), shape.end(), 1,
                                      std::multiplies<Index_t>()));
  }

  /* ---------------------------------------------------------------------- */
  size_t FFTEngineBase::size() const {
    return muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts);
  }

  /* ---------------------------------------------------------------------- */
  size_t FFTEngineBase::fourier_size() const {
    return muGrid::CcoordOps::get_size(this->nb_fourier_grid_pts);
  }

  /* ---------------------------------------------------------------------- */
  size_t FFTEngineBase::workspace_size() const {
    return this->fourier_field_collection.get_nb_pixels();
  }

  /* ---------------------------------------------------------------------- */
  const typename FFTEngineBase::Pixels_t &
  FFTEngineBase::get_real_pixels() const {
    return this->real_field_collection.get_pixels();
  }

  /* ---------------------------------------------------------------------- */
  const typename FFTEngineBase::Pixels_t &
  FFTEngineBase::get_fourier_pixels() const {
    return this->fourier_field_collection.get_pixels();
  }

  /* ---------------------------------------------------------------------- */
  const Index_t & FFTEngineBase::get_spatial_dim() const {
    return this->spatial_dimension;
  }

  /* ---------------------------------------------------------------------- */
  bool FFTEngineBase::has_plan_for(const Index_t & nb_dof_per_pixel) const {
    return static_cast<bool>(this->planned_nb_dofs.count(nb_dof_per_pixel));
  }

  /* ---------------------------------------------------------------------- */
  bool FFTEngineBase::check_real_space_field(const RealField_t & field,
                                             FFTDirection direction) const {
    if (!this->engine_has_rigid_memory_layout) {
      // If there is no requirements on the memory layout, this field is always
      // acceptable.
      return true;
    }
    return field.get_collection().has_same_memory_layout(
        this->real_field_collection);
  }

  /* ---------------------------------------------------------------------- */
  bool FFTEngineBase::check_fourier_space_field(const FourierField_t & field,
                                                FFTDirection direction) const {
    if (!this->engine_has_rigid_memory_layout) {
      // If there is no requirements on the memory layout, this field is always
      // acceptable.
      return true;
    }
    return field.get_collection().has_same_memory_layout(
        this->fourier_field_collection);
  }

  bool FFTEngineBase::check_halfcomplex_field(const RealField_t & field,
                                              FFTDirection direction) const {
    if (!this->engine_has_rigid_memory_layout) {
      // If there is no requirements on the memory layout, this field is always
      // acceptable.
      return true;
    }
    return field.get_collection().has_same_memory_layout(
        this->halfcomplex_field_collection);
  }

  /* ---------------------------------------------------------------------- */

  /* ---------------------------------------------------------------------- */
  void FFTEngineBase::compute_hcfft(const RealField_t & /*input_field*/,
                                    RealField_t & /*output_field*/) {
    throw FFTEngineError("Real to half-complex"
                         "transform not implemented in this FFTEngine");
  }

  /* ---------------------------------------------------------------------- */
  void FFTEngineBase::compute_ihcfft(const RealField_t & /*input_field*/,
                                     RealField_t & /*output_field*/) {
    throw FFTEngineError("Real to half-complex "
                         "inverse transform not implemented in this FFTEngine");
  }

  /* ---------------------------------------------------------------------- */
  void FFTEngineBase::initialise_field_collections() {
    this->real_field_collection.initialise(
        this->nb_domain_grid_pts, this->nb_subdomain_grid_pts,
        this->subdomain_locations, this->subdomain_strides);
    this->halfcomplex_field_collection.initialise(
        this->nb_domain_grid_pts, this->nb_subdomain_grid_pts,
        this->subdomain_locations, this->subdomain_strides);
    this->fourier_field_collection.initialise(
        this->nb_domain_grid_pts, this->nb_fourier_grid_pts,
        this->fourier_locations, this->fourier_strides);
  }

}  // namespace muFFT
