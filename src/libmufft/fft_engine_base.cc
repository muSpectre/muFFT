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
 * General Public License for more details.
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
 */

#include "fft_engine_base.hh"

namespace muFFT {

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim>
  FFTEngineBase<Dim>::FFTEngineBase(Ccoord resolutions, Dim_t nb_components,
                                    Communicator comm)
      : comm{comm}, subdomain_resolutions{resolutions}, subdomain_locations{},
        fourier_resolutions{
            muGrid::CcoordOps::get_hermitian_sizes(resolutions)},
        fourier_locations{}, domain_resolutions{resolutions},
        work{muGrid::make_field<Workspace_t>("work space", work_space_container,
                                             nb_components)},
        norm_factor{1. / muGrid::CcoordOps::get_size(domain_resolutions)},
        nb_components{nb_components} {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim>
  void FFTEngineBase<Dim>::initialise(FFT_PlanFlags /*plan_flags*/) {
    this->work_space_container.initialise();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim>
  size_t FFTEngineBase<Dim>::size() const {
    return muGrid::CcoordOps::get_size(this->subdomain_resolutions);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim>
  size_t FFTEngineBase<Dim>::workspace_size() const {
    return this->work_space_container.size();
  }

  template class FFTEngineBase<muGrid::twoD>;
  template class FFTEngineBase<muGrid::threeD>;

}  // namespace muFFT