/**
 * @file   solver_single_physics.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   17 Jul 2020
 *
 * @brief  Implementation for single-physics-domain base solver
 *
 * Copyright © 2020 Till Junge
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

#include "solver_single_physics.hh"

namespace muSpectre {

  const muGrid::PhysicsDomain &
  get_only_domain(std::shared_ptr<CellData> cell_data) {
    const auto & map{cell_data->get_domain_materials()};
    if (map.size() != 1) {
      throw SolverError("Single-physics solvers can only be used with cells "
                        "that contain a single physics domain");
    }
    return map.begin()->first;
  }

  /* ---------------------------------------------------------------------- */
  SolverSinglePhysics::SolverSinglePhysics(std::shared_ptr<CellData> cell_data,
                                           const muGrid::Verbosity & verbosity)
      : Parent{cell_data, verbosity}, domain{get_only_domain(cell_data)} {};

  /* ---------------------------------------------------------------------- */
  bool SolverSinglePhysics::is_mechanics() const {
    return this->domain == muGrid::PhysicsDomain::mechanics();
  }

  /* ---------------------------------------------------------------------- */
  auto SolverSinglePhysics::evaluate_stress() -> const MappedField_t & {
    return Parent::evaluate_stress(this->domain);
  }

  /* ---------------------------------------------------------------------- */
  auto SolverSinglePhysics::evaluate_stress_tangent()
      -> std::tuple<const MappedField_t &, const MappedField_t &> {
    return Parent::evaluate_stress_tangent(this->domain);
  }

}  // namespace muSpectre