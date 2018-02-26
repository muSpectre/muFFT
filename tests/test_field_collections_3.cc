/**
 * @file   test_field_collections_3.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   19 Dec 2017
 *
 * @brief  Continuation of tests from test_field_collection_2.cc, split for faster
 * compilation
 *
 * Copyright © 2017 Till Junge
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
#include <Eigen/Dense>
#include "test_field_collections_header.hh"
#include "tests/test_goodies.hh"
#include "common/tensor_algebra.hh"


namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(field_collection_tests);

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(assignment_test, Fix, iter_collections, Fix) {
    auto t4map = Fix::t4_field.get_map();
    auto t2map = Fix::t2_field.get_map();
    auto scmap = Fix::sc_field.get_map();
    auto m2map = Fix::m2_field.get_map();

    const auto t4map_val{Matrices::Isymm<Fix::mdim()>()};
    t4map = t4map_val;
    const auto t2map_val{Matrices::I2<Fix::mdim()>()};
    t2map = t2map_val;
    const Int scmap_val{1};
    scmap = scmap_val;
    Eigen::Matrix<Complex, Fix::sdim(), Fix::mdim()> m2map_val;
    m2map_val.setRandom();
    m2map = m2map_val;
    const size_t nb_pts{Fix::fc.size()};

    testGoodies::RandRange<size_t> rnd{};
    BOOST_CHECK_EQUAL((t4map[rnd.randval(0, nb_pts-1)] - t4map_val).norm(), 0.);
    BOOST_CHECK_EQUAL((t2map[rnd.randval(0, nb_pts-1)] - t2map_val).norm(), 0.);
    BOOST_CHECK_EQUAL((scmap[rnd.randval(0, nb_pts-1)] - scmap_val),        0.);
    BOOST_CHECK_EQUAL((m2map[rnd.randval(0, nb_pts-1)] - m2map_val).norm(), 0.);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Eigentest, Fix, iter_collections, Fix) {
    auto t4eigen = Fix::t4_field.eigen();
    auto t2eigen = Fix::t2_field.eigen();

    BOOST_CHECK_EQUAL(t4eigen.rows(), ipow(Fix::mdim(), 4));
    BOOST_CHECK_EQUAL(t4eigen.cols(), Fix::t4_field.size());

    using T2_t = typename Eigen::Matrix<Real, Fix::mdim(), Fix::mdim()>;
    T2_t test_mat;
    test_mat.setRandom();
    Eigen::Map<Eigen::Array<Real, ipow(Fix::mdim(), 2), 1>> test_map(test_mat.data());
    t2eigen.col(0) = test_map;

    BOOST_CHECK_EQUAL((Fix::t2_field.get_map()[0] - test_mat).norm(), 0.);


  }

  BOOST_AUTO_TEST_SUITE_END();


}  // muSpectre