/**
 * @file   intersection_volume_calculator_corkpp.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   04 June 2018
 *
 * @brief  Calculation of the intersection volume of percipitates and pixles
 *
 * Copyright © 2018 Ali Falsafi
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

#ifndef SRC_COMMON_INTERSECTION_VOLUME_CALCULATOR_CORKPP_HH_
#define SRC_COMMON_INTERSECTION_VOLUME_CALCULATOR_CORKPP_HH_

#include "cork_interface.hh"

#include <vector>
#include <fstream>
#include <math.h>

namespace muSpectre {
  // using Dim_t = int;
  // using Real = double;

  template <Dim_t DimS>
  class Correction {
   public:
    std::array<Real, 3> correct_origin(std::array<Real, DimS> array);
    std::array<Real, 3> correct_length(std::array<Real, DimS> array);
    std::vector<std::array<Real, 3>>
    correct_vector(std::vector<std::array<Real, DimS>> vector);
  };

  template <>
  class Correction<3> {
   public:
    std::array<Real, 3> correct_origin(std::array<Real, 3> array) {
      return array;
    }

    std::array<Real, 3> correct_length(std::array<Real, 3> array) {
      return array;
    }
    std::vector<std::array<Real, 3>>
    correct_vector(std::vector<std::array<Real, 3>> vertices) {
      std::vector<std::array<Real, 3>> corrected_convex_poly_vertices;
      return vertices;
    }
  };

  template <>
  class Correction<2> {
   public:
    std::vector<std::array<Real, 3>>
    correct_vector(std::vector<std::array<Real, 2>> vertices) {
      std::vector<corkpp::point_t> corrected_convex_poly_vertices;
      for (auto && vertice : vertices) {
        corrected_convex_poly_vertices.push_back({vertice[0], vertice[1], 0.0});
      }
      for (auto && vertice : vertices) {
        corrected_convex_poly_vertices.push_back({vertice[0], vertice[1], 1.0});
      }
      return corrected_convex_poly_vertices;
    }
    std::array<Real, 3> correct_origin(std::array<Real, 2> array) {
      return std::array<Real, 3>{array[0], array[1], 0.0};
    }

    std::array<Real, 3> correct_length(std::array<Real, 2> array) {
      return std::array<Real, 3>{array[0], array[1], 1.0};
    }
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  class PrecipitateIntersectBase {
   public:
    using Return_t = corkpp::VolNormStateIntersection;
    static inline std::tuple<std::vector<corkpp::point_t>,
                             std::vector<corkpp::point_t>>
    correct_dimension(std::vector<std::array<Real, DimS>> convex_poly_vertices,
                      std::array<Real, DimS> origin,
                      std::array<Real, DimS> lengths);
    static inline auto intersect_precipitate(
        std::vector<std::array<Real, DimS>> convex_poly_vertices,
        std::array<Real, DimS> origin, std::array<Real, DimS> lengths)
        -> Return_t;
  };

  /* ---------------------------------------------------------------------- */

  template <Dim_t DimS>
  std::tuple<std::vector<corkpp::point_t>, std::vector<corkpp::point_t>>
  PrecipitateIntersectBase<DimS>::correct_dimension(
      std::vector<std::array<Real, DimS>> convex_poly_vertices,
      std::array<Real, DimS> origin, std::array<Real, DimS> lengths) {
    Correction<DimS> correction;
    std::vector<corkpp::point_t> vertices_pixel{};

    std::vector<corkpp::point_t> corrected_convex_poly_vertices(
        correction.correct_vector(convex_poly_vertices));
    corkpp::point_t corrected_origin(correction.correct_origin(origin));
    corkpp::point_t corrected_lengths(correction.correct_length(lengths));

    vertices_pixel =
        corkpp::cube_vertice_maker(corrected_origin, corrected_lengths);
    return std::make_tuple(corrected_convex_poly_vertices, vertices_pixel);
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  auto PrecipitateIntersectBase<DimS>::intersect_precipitate(
      std::vector<std::array<Real, DimS>> convex_poly_vertices,
      std::array<Real, DimS> origin, std::array<Real, DimS> lengths)
      -> Return_t {
    auto && precipitate_pixel =
        correct_dimension(convex_poly_vertices, origin, lengths);
    auto && intersect = corkpp::calculate_intersection_volume_normal_state(
        std::get<0>(precipitate_pixel), std::get<1>(precipitate_pixel), DimS);
    return std::move(intersect);
  }

  /* ---------------------------------------------------------------------- */

}  // namespace muSpectre
#endif  // SRC_COMMON_INTERSECTION_VOLUME_CALCULATOR_CORKPP_HH_