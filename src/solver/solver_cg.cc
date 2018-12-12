/**
 * file   solver_cg.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Apr 2018
 *
 * @brief  implements SolverCG
 *
 * Copyright © 2018 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#include "solver/solver_cg.hh"
#include "common/communicator.hh"

#include <iomanip>
#include <sstream>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  SolverCG::SolverCG(Cell &cell, Real tol, Uint maxiter, bool verbose)
      : Parent(cell, tol, maxiter, verbose), r_k(cell.get_nb_dof()),
        p_k(cell.get_nb_dof()), Ap_k(cell.get_nb_dof()),
        x_k(cell.get_nb_dof()) {}

  /* ---------------------------------------------------------------------- */
  auto SolverCG::solve(const ConstVector_ref rhs) -> Vector_map {
    this->x_k.setZero();
    const Communicator &comm = this->cell.get_communicator();

    // Following implementation of algorithm 5.2 in Nocedal's
    // Numerical Optimization (p. 112)

    // initialisation of algorithm
    this->r_k =
        (this->cell.evaluate_projected_directional_stiffness(this->x_k) - rhs);
    this->p_k = -this->r_k;
    this->converged = false;

    Real rdr = comm.sum(this->r_k.dot(this->r_k));
    Real rhs_norm2 = comm.sum(rhs.squaredNorm());
    Real tol2 = ipow(this->tol, 2) * rhs_norm2;

    size_t count_width{};  // for output formatting in verbose case
    if (this->verbose) {
      count_width = size_t(std::log10(this->maxiter)) + 1;
    }

    for (Uint i = 0; i < this->maxiter && (rdr > tol2 || i == 0);
         ++i, ++this->counter) {
      this->Ap_k =
          this->cell.evaluate_projected_directional_stiffness(this->p_k);

      Real alpha = rdr / comm.sum(this->p_k.dot(this->Ap_k));

      this->x_k += alpha * this->p_k;
      this->r_k += alpha * this->Ap_k;

      Real new_rdr = comm.sum(this->r_k.dot(this->r_k));
      Real beta = new_rdr / rdr;
      rdr = new_rdr;

      if (this->verbose && comm.rank() == 0) {
        std::cout << "  at CG step " << std::setw(count_width) << i
                  << ": |r|/|b| = " << std::setw(15)
                  << std::sqrt(rdr / rhs_norm2) << ", cg_tol = " << this->tol
                  << std::endl;
      }

      this->p_k = -this->r_k + beta * this->p_k;
    }

    if (rdr < tol2) {
      this->converged = true;
    } else {
      std::stringstream err{};
      err << " After " << this->counter << " steps, the solver "
          << " FAILED with  |r|/|b| = " << std::setw(15)
          << std::sqrt(rdr / rhs_norm2) << ", cg_tol = " << this->tol
          << std::endl;
      throw ConvergenceError("Conjugate gradient has not converged." +
                             err.str());
    }
    return Vector_map(this->x_k.data(), this->x_k.size());
  }

}  // namespace muSpectre
