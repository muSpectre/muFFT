
/**
 * @file   fftw_engine.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Dec 2017
 *
 * @brief  FFT engine using FFTW
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

#ifndef SRC_LIBMUFFT_FFTW_ENGINE_HH_
#define SRC_LIBMUFFT_FFTW_ENGINE_HH_

#include "fft_engine_base.hh"

#include <fftw3.h>

namespace muFFT {

  /**
   * implements the `muFFT::FftEngine_Base` interface using the
   * FFTW library
   */
  class FFTWEngine : public FFTEngineBase {
   public:
    using Parent = FFTEngineBase;  //!< base class
    //! field for Fourier transform of second-order tensor
    using FourierField_t = typename Parent::FourierField_t;
    //! real-valued second-order tensor
    using RealField_t = typename Parent::RealField_t;
    //! Default constructor
    FFTWEngine() = delete;

    /**
     * @brief Constructs a FFTWEngine object with the specified parameters.
     *
     * This constructor initializes a FFTWEngine object with the given number of
     * grid points, communicator, FFT planner flags, and buffer options. The
     * constructor does not perform any FFT computations; it merely sets up the
     * object for future FFT operations.
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
     */
    FFTWEngine(const DynCcoord_t & nb_grid_pts,
               Communicator comm = Communicator(),
               const FFT_PlanFlags & plan_flags = FFT_PlanFlags::estimate,
               bool allow_temporary_buffer = true,
               bool allow_destroy_input = false);

    /**
     * @brief Constructs a FFTWEngine object with the specified parameters.
     *
     * This constructor initializes a FFTWEngine object with the given number of
     * grid points and FFT planner flags. The constructor does not perform any
     * FFT computations; it merely sets up the object for future FFT operations.
     *
     * @param nb_grid_pts A DynCcoord_t object representing the number of grid
     * points in each dimension of the global grid.
     * @param plan_flags An FFT_PlanFlags object representing the FFT planner
     * flags.
     * @param allow_temporary_buffer A boolean flag indicating whether the
     * creation of temporary buffers is allowed if the input buffer has the
     * wrong memory layout. Defaults to true.
     * @param allow_destroy_input A boolean flag indicating whether the input
     * buffers can be invalidated during the FFT. Defaults to false.
     */
    FFTWEngine(const DynCcoord_t & nb_grid_pts,
               const FFT_PlanFlags & plan_flags,
               bool allow_temporary_buffer = true,
               bool allow_destroy_input = false);

    //! Copy constructor
    FFTWEngine(const FFTWEngine & other) = delete;

    //! Move constructor
    FFTWEngine(FFTWEngine && other) = delete;

    //! Destructor
    virtual ~FFTWEngine() noexcept;

    //! Copy assignment operator
    FFTWEngine & operator=(const FFTWEngine & other) = delete;

    //! Move assignment operator
    FFTWEngine & operator=(FFTWEngine && other) = delete;

    // compute the plan, etc
    void create_plan(const Index_t & nb_dof_per_pixel) override;

    //! perform a deep copy of the engine (this should never be necessary in
    //! c++)
    std::unique_ptr<FFTEngineBase> clone() const final;

   protected:
    //! forward transform
    void compute_fft(const RealField_t & input_field,
                     FourierField_t & output_field) override;

    //! inverse transform
    void compute_ifft(const FourierField_t & input_field,
                      RealField_t & output_field) override;

    //! forward half complex transform
    void compute_hcfft(const RealField_t & input_field,
                       RealField_t & output_field) override;

    //! inverse half complex transform
    void compute_ihcfft(const RealField_t & input_field,
                        RealField_t & output_field) override;

    //! check whether Fourier-space buffer has the correct memory layout
    bool check_fourier_space_field(const FourierField_t & field,
                                   FFTDirection direction) const final;

    //! holds the plans for forward fourier transforms
    std::map<Index_t, fftw_plan> fft_plans{};
    //! holds the plans for inversefourier transforms
    std::map<Index_t, fftw_plan> ifft_plans{};

    //! holds the plans for forward half-complex fourier transforms
    std::map<Index_t, fftw_plan> hcfft_plans{};

    //! holds the plans for inverse half-complex fourier transforms
    std::map<Index_t, fftw_plan> ihcfft_plans{};

    //! global counter for total number of initialized engines
    static Index_t nb_engines;
  };

}  // namespace muFFT

#endif  // SRC_LIBMUFFT_FFTW_ENGINE_HH_
