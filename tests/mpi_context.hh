/**
 * @file   mpi_context.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   07 Mar 2018
 *
 * @brief  Singleton for initialization and tear down of MPI.
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

#ifndef TESTS_LIBMUFFT_MPI_CONTEXT_HH_
#define TESTS_LIBMUFFT_MPI_CONTEXT_HH_

#include <boost/test/unit_test.hpp>

#include <libmugrid/communicator.hh>

using muGrid::Communicator;

namespace muFFT {

  /*!
   * MPI context singleton. Initialize MPI once when needed.
   */
  class MPIContext {
   public:
    muGrid::Communicator comm, serial_comm;
    static MPIContext & get_context() {
      static MPIContext context;
      return context;
    }

   private:
    MPIContext()
#ifdef WITH_MPI
        : comm(muGrid::Communicator(MPI_COMM_WORLD)),
#else
        : comm(),
#endif
          serial_comm() {
#ifdef WITH_MPI
      MPI_Init(&boost::unit_test::framework::master_test_suite().argc,
               &boost::unit_test::framework::master_test_suite().argv);
#endif
    }
    ~MPIContext() {
#ifdef WITH_MPI
      // Wait for all processes to finish before calling finalize.
      MPI_Barrier(comm.get_mpi_comm());
      MPI_Finalize();
#endif
    }

   public:
    MPIContext(MPIContext const &) = delete;
    void operator=(MPIContext const &) = delete;
  };

}  // namespace muFFT

#endif  // TESTS_LIBMUFFT_MPI_CONTEXT_HH_
