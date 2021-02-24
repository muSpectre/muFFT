/**
 * @file bind_py_material_visco_elastic_damage_ss1.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   30 Jan 2020
 *
 * @brief python binding of MaterialLinearViscoElasticDeviatoricDamageSmallStrai
 *
 * Copyright © 2020 Ali Falsafi
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

#include "common/muSpectre_common.hh"
#include "materials/material_visco_elastic_damage_ss1.hh"
#include "cell/cell.hh"
#include "cell/cell_data.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <string>

using muSpectre::Index_t;
using muSpectre::Real;
using pybind11::literals::operator""_a;
namespace py = pybind11;

/**
 * python binding for the material with viscoelasticity for the deviatoric
 * loading and damage
 */
template <Index_t Dim>
void add_material_visco_elastic_damage_ss1_helper(
    py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialViscoElasticDamageSS1_" << Dim
              << "d";
  const auto name{name_stream.str()};

  using Mat_t =
      muSpectre::MaterialViscoElasticDamageSS1<Dim>;
  using Cell_t = muSpectre::Cell;
  using CellData_t = muSpectre::CellData;

  py::class_<Mat_t, muSpectre::MaterialBase, std::shared_ptr<Mat_t>>(
      mod, name.c_str())
      .def_static(
          "make",
          [](std::shared_ptr<Cell_t> cell, const std::string & name,
             const Real & young_inf, const Real & young_v, const Real & eta_v,
             const Real & poisson, const Real & kappa, const Real & alpha,
             const Real & beta, const Real & dt = 0.0) -> Mat_t & {
            return Mat_t::make(cell, name, young_inf, young_v, eta_v, poisson,
                               kappa, alpha, beta, dt);
          },
          "cell"_a, "name"_a, "YoungModulusInf"_a, "YoungModulusV"_a, "EtaV"_a,
          "PoissonRatio"_a, "Kappa"_a, "Alpha"_a, "Beta"_a, "dt"_a,
          py::return_value_policy::reference_internal)
      .def_static(
          "make",
          [](std::shared_ptr<CellData_t> cell, const std::string & name,
             const Real & young_inf, const Real & young_v, const Real & eta_v,
             const Real & poisson, const Real & kappa, const Real & alpha,
             const Real & beta, const Real & dt = 0.0) -> Mat_t & {
            return Mat_t::make(cell, name, young_inf, young_v, eta_v, poisson,
                               kappa, alpha, beta, dt);
          },
          "cell"_a, "name"_a, "YoungModulusInf"_a, "YoungModulusV"_a, "EtaV"_a,
          "PoissonRatio"_a, "Kappa"_a, "Alpha"_a, "Beta"_a, "dt"_a,
          py::return_value_policy::reference_internal)
      .def_static(
          "make_evaluator",
          [](const Real & young_inf, const Real & young_v, const Real & eta_v,
             const Real & poisson, const Real & kappa, const Real & alpha,
             const Real & beta, const Real & dt = 0.0) {
            return Mat_t::make_evaluator(young_inf, young_v, eta_v, poisson,
                                         kappa, alpha, beta, dt);
          },
          "YoungModulusInf"_a, "YoungModulusV"_a, "EtaV"_a, "PoissonRatio"_a,
          "Kappa"_a, "Alpha"_a, "Beta"_a, "dt"_a);
}

template void
add_material_visco_elastic_damage_ss1_helper<
    muSpectre::twoD>(py::module &);
template void
add_material_visco_elastic_damage_ss1_helper<
    muSpectre::threeD>(py::module &);
