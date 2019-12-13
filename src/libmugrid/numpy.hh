/**
 * @file   numpy.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   02 Dec 2019
 *
 * @brief  Convenience function for working with (pybind11's) numpy arrays
 *
 * Copyright © 2018 Till Junge
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
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

#ifndef SRC_LIBMUGRID_NUMPY_HH_
#define SRC_LIBMUGRID_NUMPY_HH_

#include <algorithm>

#include <pybind11/numpy.h>

#include "nfield_typed.hh"
#include "nfield_collection_global.hh"

namespace muGrid {

/**
 * base class for numpy related exceptions
 */
class NumpyError : public std::runtime_error {
 public:
  //! constructor
  explicit NumpyError(const std::string & what)
      : std::runtime_error(what) {}
  //! constructor
  explicit NumpyError(const char * what) : std::runtime_error(what) {}
};

/**
 * Wrap a pybind11::array into a WrappedNField and check the shape of the
 * array
 */
template <typename T, class Collection_t = GlobalNFieldCollection>
class NumpyProxy {
 public:
  /**
   * Construct a NumpyProxy given that we only know the number of components
   * of the field. The constructor will complain if the grid dimension differs
   * but will wrap any field whose number of components match. For example,
   * a 3x3 grid with 8 components could look like this:
   *    1. (8, 3, 3)
   *    2. (2, 4, 3, 3)
   *    3. (2, 2, 2, 3, 3)
   * The method `get_components_shape` return the shape of the component part
   * of the field in this case. For the above examples, it would return:
   *    1. (8,)
   *    2. (2, 4)
   *    3. (2, 2, 2)
   * Note that a field with a single component can be passed either with a
   * shaping having leading dimension of one or without any leading dimension.
   * In the latter case, `get_component_shape` will return a vector of size 0.
   * The same applies for fields with a single quadrature point, whose
   * dimension can be omitted. In general, the shape of the field needs to
   * look like this:
   *    (component_1, component:2, quad_pt, grid_x, grid_y, grid_z)
   * where the number of components and grid indices can be arbitrary.
   */
  NumpyProxy(DynCcoord_t nb_subdomain_grid_pts,
             DynCcoord_t subdomain_locations,
             Dim_t nb_components,
             pybind11::array_t<T, pybind11::array::f_style> array)
      : collection{nb_subdomain_grid_pts.get_dim(),
                   OneQuadPt,
                   nb_subdomain_grid_pts,
                   subdomain_locations},
        field{"proxy_field",
              collection,
              nb_components,
              static_cast<size_t>(array.request().size),
              static_cast<T*>(array.request().ptr)},
        quad_pt_shape{0},
        components_shape{} {
    // Note: There is a check on the global array size in the constructor of
    // WrappedNField, which will fail before the sanity checks below.
    Dim_t dim = nb_subdomain_grid_pts.get_dim();
    pybind11::buffer_info buffer = array.request();
    if (!std::equal(nb_subdomain_grid_pts.begin(), nb_subdomain_grid_pts.end(),
                    buffer.shape.end() - dim)) {
      std::stringstream s;
      s << "The numpy array has shape " << buffer.shape << ", but the muGrid "
        << "field reports a grid of " << this->field.get_pixels_shape()
        << " pixels. The numpy array must equal the grid size in its last "
        << "dimensions.";
      throw NumpyError(s.str());
    }
    Dim_t nb_array_components = 1;
    for (auto n = buffer.shape.begin(); n != buffer.shape.end() - dim; ++n) {
      this->components_shape.push_back(*n);
      nb_array_components *= *n;
    }
    if (nb_array_components != nb_components) {
      std::stringstream s;
      s << "The numpy array has shape " << buffer.shape << ", but the muGrid "
        << "field reports " << nb_components << " components. The numpy array "
        << "must equal the number of components in its first dimensions.";
      throw NumpyError(s.str());
    }
  }

  /**
   * Construct a NumpyProxy given that we know the shape of the leading
   * component indices. The constructor will complain if both the grid
   * dimensions and the component dimensions differ. `get_component_shape`
   * returns exactly the shape passed to this constructor.
   *
   * In general, the shape of the field needs to look like this:
   *    (component_1, component:2, quad_pt, grid_x, grid_y, grid_z)
   * where the number of components and grid indices can be arbitrary. The
   * quad_pt dimension can be omitted if there is only a single quad_pt.
   */
  NumpyProxy(DynCcoord_t nb_subdomain_grid_pts,
             DynCcoord_t subdomain_locations,
             Dim_t nb_quad_pts, std::vector<Dim_t> components_shape,
             pybind11::array_t<T, pybind11::array::f_style> array)
      : collection{nb_subdomain_grid_pts.get_dim(),
                   nb_quad_pts,
                   nb_subdomain_grid_pts,
                   subdomain_locations},
        field{"proxy_field",
              collection,
              std::accumulate(components_shape.begin(), components_shape.end(),
                              1, std::multiplies<Dim_t>()),
              static_cast<size_t>(array.request().size),
              static_cast<T*>(array.request().ptr)},
        quad_pt_shape{nb_quad_pts},
        components_shape{components_shape} {
    // Note: There is a check on the global array size in the constructor of
    // WrappedNField, which will fail before the sanity checks below.
    Dim_t dim = nb_subdomain_grid_pts.get_dim();
    pybind11::buffer_info buffer = array.request();
    bool shape_matches = false;
    if (dim + components_shape.size() + 1 == buffer.shape.size()) {
      shape_matches =
          std::equal(nb_subdomain_grid_pts.begin(), nb_subdomain_grid_pts.end(),
                     buffer.shape.end() - dim) &&
          nb_quad_pts == buffer.shape[components_shape.size()] &&
          std::equal(components_shape.begin(), components_shape.end(),
                      buffer.shape.begin());
    } else if (dim + components_shape.size() == buffer.shape.size()) {
      // For a field with a single quad point, we can omit that dimension.
      shape_matches =
          std::equal(nb_subdomain_grid_pts.begin(), nb_subdomain_grid_pts.end(),
                     buffer.shape.end() - dim) &&
          nb_quad_pts == 1 &&
          std::equal(components_shape.begin(), components_shape.end(),
                      buffer.shape.begin());
      this->quad_pt_shape = 0;
    }
    if (!shape_matches) {
      std::stringstream s;
      s << "The numpy array has shape " << buffer.shape << ", but the muGrid "
        << "field reports a grid of " << this->field.get_pixels_shape()
        << " pixels with " << nb_quad_pts << " quadrature "
        << (nb_quad_pts == 1 ? "point" : "points") << " holding a quantity of "
        << "shape " << components_shape << ".";
      throw NumpyError(s.str());
    }
  }

  /**
   * move constructor
   */
  NumpyProxy(NumpyProxy && other) = default;

  WrappedNField<T> & get_field() { return this->field; }

  const std::vector<Dim_t> & get_components_shape() const {
    return this->components_shape;
  }

  std::vector<Dim_t> get_components_and_quad_pt_shape() const {
    std::vector<Dim_t> shape;
    for (auto && n : this->components_shape) {
      shape.push_back(n);
    }
    if (this->quad_pt_shape > 0) {
      shape.push_back(this->quad_pt_shape);
    }
    return shape;
  }

 protected:
  Collection_t collection;
  WrappedNField<T> field;
  Dim_t quad_pt_shape;                  //! number of quad pts, omit if zero
  std::vector<Dim_t> components_shape;  //! shape of the components
};

/* Copy a numpy array into an existing field while checking the shapes */
template <typename T>
std::vector<Dim_t>
numpy_copy(const TypedNFieldBase<T> & field,
           pybind11::array_t<T, pybind11::array::f_style> array) {
  std::vector<Dim_t> pixels_shape{field.get_pixels_shape()};
  pybind11::buffer_info buffer = array.request();
  if (!std::equal(pixels_shape.begin(), pixels_shape.end(),
                  buffer.shape.end() - pixels_shape.size())) {
    std::stringstream s;
    s << "The numpy array has shape " << buffer.shape << ", but the muGrid "
      << "field reports a grid of " << pixels_shape << " pixels. The numpy "
      << "array must equal the grid size in its last dimensions.";
    throw NumpyError(s.str());
  }
  Dim_t nb_array_components = 1;
  std::vector<Dim_t> components_shape;
  for (auto n = buffer.shape.begin();
       n != buffer.shape.end() - pixels_shape.size(); ++n) {
    components_shape.push_back(*n);
    nb_array_components *= *n;
  }
  if (nb_array_components != field.get_nb_components()) {
    std::stringstream s;
    s << "The numpy array has shape " << buffer.shape << ", but the muGrid "
      << "field reports " << field.get_nb_components() << " components per "
      << "pixel. The numpy array must equal the number of components in its "
      << "first dimensions.";
    throw NumpyError(s.str());
  }

  std::copy(static_cast<T*>(buffer.ptr),
            static_cast<T*>(buffer.ptr) + buffer.size,
            field.data());

  return components_shape;
}

/* Wrap a field into a numpy array, without copying the data */
template <typename T>
pybind11::array_t<T, pybind11::array::f_style>
numpy_wrap(const TypedNFieldBase<T> & field,
           std::vector<Dim_t> components_shape = std::vector<Dim_t>{}) {
  std::vector<Dim_t> shape{};
  if (components_shape.size() != 0) {
    if (field.get_nb_components() != std::accumulate(
        components_shape.begin(), components_shape.end(), 1,
        std::multiplies<Dim_t>())) {
      std::stringstream s;
      s << "Unable to wrap field with " << field.get_nb_components()
        << " components into a numpy array with " << components_shape
        << " components.";
      throw NumpyError(s.str());
    }
    shape = components_shape;
  } else {
    shape.push_back(field.get_nb_components());
  }
  for (auto && n : field.get_pixels_shape()) {
    shape.push_back(n);
  }
  return pybind11::array_t<T, pybind11::array::f_style>(
      shape, field.data(), pybind11::capsule([]() {}));
}

/* Turn any type that can be enumerated into a tuple */
template <typename T>
pybind11::tuple to_tuple(T a) {
  pybind11::tuple t(a.get_dim());
  ssize_t i = 0;
  for (auto && v : a) {
    t[i] = v;
    i++;
  }
  return t;
}

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_NUMPY_HH_