/**
 * @file   test_state_nfields.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   21 Aug 2019
 *
 * @brief  Tests for state fields
 *
 * Copyright © 2019 Till Junge
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#include "tests.hh"
#include "libmugrid/state_nfield.hh"
#include "libmugrid/nfield_typed.hh"
#include "libmugrid/nfield_collection_local.hh"

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(state_nfield_tests);

  struct LocalNFieldBasicFixture {
    LocalNFieldCollection fc{Unknown, Unknown};
  };

  struct LocalNFieldBasicFixtureFilled : public LocalNFieldBasicFixture {
    static constexpr Dim_t NbMemory() { return 3; }
    static constexpr Dim_t NbComponents() { return 21; }

    LocalNFieldBasicFixtureFilled()
        : state_field{fc.register_state_field<Real>(
              "test", NbMemory(), NbComponents())} {
      this->fc.add_pixel(4);
      this->fc.add_pixel(8);
      this->fc.set_nb_quad(2);
      this->fc.initialise();
    }
    TypedStateNField<Real> & state_field;
  };

  BOOST_FIXTURE_TEST_CASE(construction_test, LocalNFieldBasicFixture) {
    constexpr Dim_t NbMemory{1}, NbComponents{21};
    auto & state_field{fc.register_state_field<Real>(
        "test", NbMemory, NbComponents)};

    state_field.current();
    BOOST_CHECK(fc.state_field_exists("test"));

    BOOST_CHECK_THROW(fc.register_state_field<Real>(
                          "test", NbMemory, NbComponents),
                      NFieldCollectionError);
  }

  BOOST_FIXTURE_TEST_CASE(cycle_test, LocalNFieldBasicFixtureFilled) {
    auto & current_field{state_field.current()};
    current_field.eigen_quad_pt().setZero();
    for (Dim_t i{0}; i < NbMemory(); ++i) {
      state_field.cycle();
      auto map{state_field.current().eigen_quad_pt()};
      map.setOnes();
      map *= (i + 1);
    }

    BOOST_CHECK_EQUAL(state_field.current().eigen_vec()(0), NbMemory());

    for (Dim_t i{0}; i < NbMemory(); ++i) {
      BOOST_CHECK_EQUAL(state_field.old(i + 1).eigen_vec()(0),
                        NbMemory() - 1 - i);
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
