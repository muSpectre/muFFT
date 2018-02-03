/**
* @file   solver_cg.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Dec 2017
 *
 * @brief  Implementation of cg solver
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

#include "solver/solver_cg.hh"
#include "solver/solver_error.hh"

#include <iomanip>
#include <cmath>
#include <sstream>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  SolverCG<DimS, DimM>::SolverCG(Sys_t& sys, Real tol, Uint maxiter,
                                 bool verbose)
    :Parent(sys, tol, maxiter, verbose),
     r_k{make_field<Field_t>("residual r_k", this->collection)},
     p_k{make_field<Field_t>("search direction r_k", this->collection)},
     Ap_k{make_field<Field_t>("Effect of tangent A*p_k", this->collection)}
  {}


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void SolverCG<DimS, DimM>::solve(const Field_t & rhs,
                                   Field_t & x_f) {
    x_f.eigenvec() = this-> solve(rhs.eigenvec(), x_f.eigenvec());
  };

  //----------------------------------------------------------------------------//
  template <Dim_t DimS, Dim_t DimM>
  typename SolverCG<DimS, DimM>::SolvVectorOut
  SolverCG<DimS, DimM>::solve(const SolvVectorInC rhs, SolvVectorIn x_0) {
    // Following implementation of algorithm 5.2 in Nocedal's Numerical Optimization (p. 112)

    auto r = this->r_k.eigen();
    auto p = this->p_k.eigen();
    auto Ap = this->Ap_k.eigen();
    auto x = typename Field_t::EigenMap(x_0.data(), r.rows(), r.cols());

    // initialisation of algo
    r = this->sys.directional_stiffness_with_copy(x);

    r -= typename Field_t::ConstEigenMap(rhs.data(), r.rows(), r.cols());
    p = -r;

    this->converged = false;
    Real rdr = (r*r).sum();
    Real rhs_norm2 = rhs.squaredNorm();
    Real tol2 = ipow(this->tol,2)*rhs_norm2;

    size_t count_width{}; // for output formatting in verbose case
    if (this->verbose) {
      count_width = size_t(std::log10(this->maxiter))+1;
    }

    for (Uint i = 0;
         i < this->maxiter && (rdr > tol2 || i == 0);
         ++i, ++this->counter) {
      Ap = this->sys.directional_stiffness_with_copy(p);

      Real alpha = rdr/(p*Ap).sum();

      x += alpha * p;
      r += alpha * Ap;

      Real new_rdr = (r*r).sum();
      Real beta = new_rdr/rdr;
      rdr = new_rdr;

      if (this->verbose) {
        std::cout << "  at CG step " << std::setw(count_width) << i
                  << ": |r|/|b| = " << std::setw(15) << std::sqrt(rdr/rhs_norm2)
                  << ", cg_tol = " << this->tol << std::endl;
      }

      p = -r+beta*p;
    }
    if (rdr < tol2) {
      this->converged=true;
    } else {
      Real * a = nullptr;
      std::stringstream err {};
      err << " After " << this->counter << " steps, the solver "
          << " FAILED with  |r|/|b| = "
          << std::setw(15) << std::sqrt(rdr/rhs_norm2)
          << ", cg_tol = " << this->tol << *a << std::endl;
      throw ConvergenceError("Conjugate gradient has not converged." + err.str());
    }
    return x_0;
  }


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename SolverCG<DimS, DimM>::Tg_req_t
  SolverCG<DimS, DimM>::get_tangent_req() const {
    return tangent_requirement;
  }

  template class SolverCG<twoD, twoD>;
  //template class SolverCG<twoD, threeD>;
  template class SolverCG<threeD, threeD>;
}  // muSpectre
