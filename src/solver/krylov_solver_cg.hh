/**
 * @file   krylov_solver_cg.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Apr 2018
 *
 * @brief  class fo a simple implementation of a conjugate gradient solver.
 *         This follows algorithm 5.2 in Nocedal's Numerical Optimization
 *         (p 112)
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
 * Lesser General Public License for more details.
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
 *
 */

#ifndef SRC_SOLVER_KRYLOV_SOLVER_CG_HH_
#define SRC_SOLVER_KRYLOV_SOLVER_CG_HH_

#include "solver/krylov_solver_base.hh"

namespace muSpectre {

  /**
   * implements the `muSpectre::KrylovSolverBase` interface using a
   * conjugate gradient solver. This particular class is useful for
   * trouble shooting, as it can be made very verbose. The solver complains if
   * it finds a search direction in which the problem is not convex
   * (=has no minimum).
   */
  class KrylovSolverCG : public KrylovSolverBase {
   public:
    using Parent = KrylovSolverBase;  //!< standard short-hand for base class
    //! for storage of fields
    using Vector_t = typename Parent::Vector_t;
    //! Input vector for solvers
    using Vector_ref = typename Parent::Vector_ref;
    //! Input vector for solvers
    using ConstVector_ref = typename Parent::ConstVector_ref;
    //! Output vector for solvers
    using Vector_map = typename Parent::Vector_map;

    //! Default constructor
    KrylovSolverCG() = delete;

    //! Copy constructor
    KrylovSolverCG(const KrylovSolverCG & other) = delete;

    /**
     * Constructor takes a sytem matrix, tolerance, max number of iterations
     * and verbosity flag as input
     */
    KrylovSolverCG(std::shared_ptr<MatrixAdaptable> matrix, const Real & tol,
                   const Uint & maxiter,
                   const Verbosity & verbose = Verbosity::Silent);

    /**
     * Constructor without matrix adaptable. The adaptable has to be supplied
     * using KrylovSolverBase::set_matrix(...) before initialisation for this
     * solver to be usable
     */
    KrylovSolverCG(const Real & tol, const Uint & maxiter,
                   const Verbosity & verbose = Verbosity::Silent);

    //! Move constructor
    KrylovSolverCG(KrylovSolverCG && other) = default;

    //! Destructor
    virtual ~KrylovSolverCG() = default;

    //! Copy assignment operator
    KrylovSolverCG & operator=(const KrylovSolverCG & other) = delete;

    //! Move assignment operator
    KrylovSolverCG & operator=(KrylovSolverCG && other) = delete;

    //! initialisation does not need to do anything in this case
    void initialise() final{};

    //! set the matrix
    void set_matrix(std::shared_ptr<MatrixAdaptable> matrix_adaptable) final;

    //! set the matrix
    void set_matrix(std::weak_ptr<MatrixAdaptable> matrix_adaptable) final;

    //! returns the solver's name
    std::string get_name() const final;

    //! the actual solver
    Vector_map solve(const ConstVector_ref rhs) final;

   protected:
    // to be called in set_matrix
    void set_internal_arrays();
    Vector_t r_k;   //!< residual
    Vector_t p_k;   //!< search direction
    Vector_t Ap_k;  //!< directional stiffness
    Vector_t x_k;   //!< current solution
  };

}  // namespace muSpectre

#endif  // SRC_SOLVER_KRYLOV_SOLVER_CG_HH_
