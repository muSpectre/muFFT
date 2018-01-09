/**
 * file   bind_py_material.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   09 Jan 2018
 *
 * @brief  python bindings for µSpectre's materials
 *
 * @section LICENCE
 *
 * Copyright © 2018 Till Junge
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

#include "common/common.hh"
#include "materials/material_hyper_elastic1.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <string>

using namespace muSpectre;
namespace py = pybind11;
using namespace pybind11::literals;

/**
 * python binding for the optionally objective form of Hooke's law
 */
template <Dim_t dim>
void add_material_hyper_elastic_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialHooke" << dim << 'd';
  auto && name {name_stream.str().c_str()};

  py::class_<MaterialHyperElastic1<dim, dim>>(mod, name)
    .def(py::init<std::string, Real, Real>(), "name"_a, "Young"_a, "Poisson"_a);
}

template <Dim_t dim>
void add_material_helper(py::module & mod) {
  add_material_hyper_elastic_helper<dim>(mod);
}

void add_material(py::module & mod) {
  add_material_helper<twoD  >(mod);
  add_material_helper<threeD>(mod);
}

PYBIND11_PLUGIN(material) {
  (py::object) py::module::import("common");
  (py::object) py::module::import("system");

  py::module mod("material", "bindings for constitutive laws");

  add_material(mod);

  return mod.ptr();
}
