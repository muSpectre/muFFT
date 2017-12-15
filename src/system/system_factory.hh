/**
 * file   system_factory.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   15 Dec 2017
 *
 * @brief  System factories to help create systems with ease
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef SYSTEM_FACTORY_H
#define SYSTEM_FACTORY_H

#include <memory>

#include "common/common.hh"
#include "common/ccoord_operations.hh"
#include "system/system_base.hh"
#include "fft/projection_finite_strain_fast.hh"
#include "fft/projection_finite_strain.hh"
#include "fft/fftw_engine.hh"

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM,
            typename System=SystemBase<DimS, DimM>,
            typename FFT_Engine=FFTW_Engine<DimS, DimM>,
            typename Projection=ProjectionFiniteStrainFast<DimS, DimM>>
  inline
  System make_system(Ccoord_t<DimS> resolutions,
                     Rcoord_t<DimS> lengths=CcoordOps::get_cube<DimS>(1.)) {
    auto && fft_ptr{std::make_unique<FFT_Engine>(resolutions, lengths)};
    auto && proj_ptr{std::make_unique<Projection>(std::move(fft_ptr))};
    return System{std::move(proj_ptr)};
  }

}  // muSpectre

#endif /* SYSTEM_FACTORY_H */

