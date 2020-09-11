/**
 * @file   communicator.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   02 Oct 2019
 *
 * @brief  implementation for mpi abstraction layer
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
 * Lesser General Public License for more details.
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
 *
 */

#include <sstream>

#include "exception.hh"

#include "communicator.hh"

namespace muGrid {

  Communicator::Communicator(MPI_Comm comm) : comm{comm} {}

  Communicator::Communicator(const Communicator & other)
      : comm(other.comm) {}

  Communicator::~Communicator() {}

  Communicator & Communicator::operator=(const Communicator &other) {
    this->comm = other.comm;
    return *this;
  }

  //! sum reduction on EigenMatrix types
  template <typename T>
  auto Communicator::sum_mat(const Eigen::Ref<DynMatrix_t<T>> & arg) const
      -> DynMatrix_t<T> {
    if (this->comm == MPI_COMM_NULL)
      return arg;
    DynMatrix_t<T> res(arg.rows(), arg.cols());
    res.setZero();
    const auto count{arg.size()};
    MPI_Allreduce(arg.data(), res.data(), count, mpi_type<T>(), MPI_SUM,
                  this->comm);
    return res;
  }

  template auto
  Communicator::sum_mat(const Eigen::Ref<DynMatrix_t<Real>> &) const
      -> DynMatrix_t<Real>;
  template auto
  Communicator::sum_mat(const Eigen::Ref<DynMatrix_t<Int>> &) const
      -> DynMatrix_t<int>;
  template auto
  Communicator::sum_mat(const Eigen::Ref<DynMatrix_t<Uint>> &) const
      -> DynMatrix_t<Uint>;
  template auto
  Communicator::sum_mat(const Eigen::Ref<DynMatrix_t<Complex>> &) const
      -> DynMatrix_t<Complex>;

  //! cumulative sum on scalars
  template auto Communicator::cumulative_sum(const Int &) const -> Int;
  template auto Communicator::cumulative_sum(const Real &) const -> Real;
  template auto Communicator::cumulative_sum(const Uint &) const -> Uint;
  template auto Communicator::cumulative_sum(const Index_t &) const -> Index_t;
  template auto Communicator::cumulative_sum(const Complex &) const -> Complex;

  //! gather on EigenMatrix types
  template <typename T>
  auto Communicator::gather(const Eigen::Ref<DynMatrix_t<T>> & arg) const
      -> DynMatrix_t<T> {
    if (this->comm == MPI_COMM_NULL)
      return arg;
    Index_t send_buf_size(arg.size());

    int comm_size = this->size();
    std::vector<int> arg_sizes{comm_size};
    auto message{MPI_Allgather(&send_buf_size, 1, mpi_type<int>(),
                               arg_sizes.data(), 1, mpi_type<int>(),
                               this->comm)};
    if (message != 0) {
      std::stringstream error{};
      error << "MPI_Allgather failed with " << message << " on rank "
            << this->rank();
      throw RuntimeError(error.str());
    }

    std::vector<int> displs{comm_size};
    displs[0] = 0;
    for (auto i = 0; i < comm_size - 1; ++i) {
      displs[i + 1] = displs[i] + arg_sizes[i];
    }

    int nb_entries = 0;
    for (auto i = 0; i < comm_size; ++i) {
      nb_entries += arg_sizes[i];
    }

    DynMatrix_t<T> res(arg.rows(), nb_entries / arg.rows());
    res.setZero();

    message =
        MPI_Allgatherv(arg.data(), send_buf_size, mpi_type<T>(), res.data(),
                       arg_sizes.data(), displs.data(), mpi_type<T>(),
                       this->comm);
    if (message != 0) {
      std::stringstream error{};
      error << "MPI_Allgatherv failed with " << message << " on rank "
            << this->rank();
      throw RuntimeError(error.str());
    }
    return res;
  }

  template auto
  Communicator::gather(const Eigen::Ref<DynMatrix_t<Real>> &) const
      -> DynMatrix_t<Real>;
  template auto
  Communicator::gather(const Eigen::Ref<DynMatrix_t<Int>> &) const
      -> DynMatrix_t<int>;
  template auto
  Communicator::gather(const Eigen::Ref<DynMatrix_t<Uint>> &) const
      -> DynMatrix_t<Uint>;
  template auto
  Communicator::gather(const Eigen::Ref<DynMatrix_t<Complex>> &) const
      -> DynMatrix_t<Complex>;

}  // namespace muGrid
