/**
 * @file   ncell.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   13 Sep 2019
 *
 * @brief  Class for the representation of a homogenisation problem in µSpectre
 *
 * Copyright © 2019 Till Junge
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

#ifndef SRC_CELL_NCELL_HH_
#define SRC_CELL_NCELL_HH_

#include "common/muSpectre_common.hh"
#include "materials/material_base.hh"
#include "projection/projection_base.hh"

#include <libmugrid/ccoord_operations.hh>

#include <memory>
namespace muSpectre {
  /**
   * Cell adaptors implement the matrix-vector multiplication and
   * allow the system to be used like a sparse matrix in
   * conjugate-gradient-type solvers
   */
  template <class Cell>
  class CellAdaptor;

  /**
   * Base class for the representation of a homogenisatonion problem in
   * µSpectre. The `muSpectre::NCell` holds the global strain, stress and
   * (optionally) tangent moduli fields of the problem, maintains the list of
   * materials present, as well as the projection operator.
   */
  class NCell {
   public:
    //! materials handled through `std::unique_ptr`s
    using Material_ptr = std::unique_ptr<MaterialBase>;
    //! projections handled through `std::unique_ptr`s
    using Projection_ptr = std::unique_ptr<ProjectionBase>;

    //! short-hand for matrices
    using Matrix_t = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

    //! ref to constant vector
    using Eigen_cmap = muGrid::RealNField::Eigen_cmap;
    //! ref to  vector
    using Eigen_map = muGrid::RealNField::Eigen_map;

    //! Ref to input/output vector
    using EigenVec_t = Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, 1>>;

    //! Ref to input vector
    using EigenCVec_t =
        Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>>;

    //! adaptor to represent the cell as an Eigen sparse matrix
    using Adaptor = CellAdaptor<NCell>;

    //! Deleted default constructor
    NCell() = delete;

    //! Constructor from a projection operator
    explicit NCell(Projection_ptr projection);

    //! Copy constructor
    NCell(const NCell & other) = delete;

    //! Move constructor
    NCell(NCell && other) = default;

    //! Destructor
    virtual ~NCell() = default;

    //! Copy assignment operator
    NCell & operator=(const NCell & other) = delete;

    //! Move assignment operator
    NCell & operator=(NCell && other) = delete;

    //! for handling double initialisations right
    bool is_initialised() const;

    //! returns the number of degrees of freedom in the cell
    Dim_t get_nb_dof() const;

    //! number of pixels on this processor
    size_t get_nb_pixels() const;

    //! return the communicator object
    const muFFT::Communicator & get_communicator() const;

    /**
     * formulation is hard set by the choice of the projection class
     */
    const Formulation & get_formulation() const;

    /**
     * returns the material dimension of the problem
     */
    Dim_t get_material_dim() const;

    /**
     * set uniform strain (typically used to initialise problems
     */
    void set_uniform_strain(const Eigen::Ref<const Matrix_t> &);

    /**
     * add a new material to the cell
     */
    MaterialBase & add_material(Material_ptr mat);

    //! get a sparse matrix view on the cell
    Adaptor get_adaptor();

    /**
     * freezes all the history variables of the materials
     */
    void save_history_variables();

    /**
     * returns the number of rows and cols for the strain matrix type
     * (for full storage, the strain is stored in material_dim ×
     * material_dim matrices, but in symmetric storage, it is a column
     * vector)
     */
    std::array<Dim_t, 2> get_strain_shape() const;

    /**
     * returns the number of components for the strain matrix type
     * (for full storage, the strain is stored in material_dim ×
     * material_dim matrices, but in symmetric storage, it is a column
     * vector)
     */
    Dim_t get_strain_size() const;

    //! return the spatial dimension of the discretisation grid
    const Dim_t & get_spatial_dim() const;

    //! return the number of quadrature points stored per pixel
    const Dim_t & get_nb_quad() const;

    //! makes sure every pixel has been assigned to exactly one material
    void check_material_coverage() const;

    //! initialise the projection, the materials and the global fields
    void
    initialise(muFFT::FFT_PlanFlags flags = muFFT::FFT_PlanFlags::estimate);

    //! return a const reference to the grids pixels iterator
    const muGrid::CcoordOps::DynamicPixels & get_pixels() const;

    /**
     * return an iterable proxy to this cell's field collection, iterable by
     * quadrature point
     */
    muGrid::NFieldCollection::IndexIterable get_quad_pt_indices() const;

    /**
     * return an iterable proxy to this cell's field collection, iterable by
     * pixel
     */
    muGrid::NFieldCollection::PixelIndexIterable get_pixel_indices() const;

    //! return a reference to the cell's strain field
    muGrid::RealNField & get_strain();

    //! return a const reference to the cell's stress field
    const muGrid::RealNField & get_stress() const;

    //! return a const reference to the cell's field of tangent moduli
    const muGrid::RealNField & get_tangent(bool do_create = false);

    /**
     * evaluates and returns the stress for the currently set strain
     */
    const muGrid::RealNField & evaluate_stress();

    /**
     * evaluates and returns the stress for the currently set strain
     */
    Eigen_cmap evaluate_stress_eigen();

    /**
     * evaluates and returns the stress and tangent moduli for the currently set
     * strain
     */
    std::tuple<const muGrid::RealNField &, const muGrid::RealNField &>
    evaluate_stress_tangent();

    /**
     * evaluates and returns the stress and tangent moduli for the currently set
     * strain
     */
    std::tuple<const Eigen_cmap, const Eigen_cmap>
    evaluate_stress_tangent_eigen();

    /**
     * collect the real-valued fields of name `unique_name` of each material in
     * the cell and write their values into a global field of same type and name
     */
    muGrid::RealNField &
    globalise_real_internal_field(const std::string & unique_name);

    /**
     * collect the integer-valued fields of name `unique_name` of each material
     * in the cell and write their values into a global field of same type and
     * name
     */
    muGrid::IntNField &
    globalise_int_internal_field(const std::string & unique_name);

    /**
     * collect the unsigned integer-valued fields of name `unique_name` of each
     * material in the cell and write their values into a global field of same
     * type and name
     */
    muGrid::UintNField &
    globalise_uint_internal_field(const std::string & unique_name);

    /**
     * collect the complex-valued fields of name `unique_name` of each material
     * in the cell and write their values into a global field of same type and
     * name
     */
    muGrid::ComplexNField &
    globalise_complex_internal_field(const std::string & unique_name);

    //! return a reference to the cell's global fields
    muGrid::GlobalNFieldCollection & get_fields();

    //! apply the cell's projection operator to field `field` (i.e., return G:f)
    void apply_projection(muGrid::TypedNFieldBase<Real> & field);
    /**
     * evaluates the directional and projected stiffness (this
     * corresponds to G:K:δF (note the negative sign in de Geus 2017,
     * http://dx.doi.org/10.1016/j.cma.2016.12.032).
     */
    void evaluate_projected_directional_stiffness(
        const muGrid::TypedNFieldBase<Real> & delta_strain,
        muGrid::TypedNFieldBase<Real> & del_stress);

    /**
     * evaluates the directional and projected stiffness (this
     * corresponds to G:K:δF (note the negative sign in de Geus 2017,
     * http://dx.doi.org/10.1016/j.cma.2016.12.032). and then adds it do the
     * values already in del_stress, scaled by alpha (i.e., del_stress +=
     * alpha*Q:K:δStrain. This function should not be used directly, as it does
     * absolutely no input checking. Rather, it is meant to be called by the
     * scaleAndAddTo function in the CellAdaptor
     */
    void add_projected_directional_stiffness(EigenCVec_t delta_strain,
                                             const Real & alpha,
                                             EigenVec_t del_stress);

    //! transitional function, use discouraged
    SplitCell get_splitness() const { return SplitCell::no; }

    //! return a const ref to the projection implementation
    const ProjectionBase & get_projection() const;

   protected:
    //! statically dimensioned worker for evaluating the tangent operator
    template <Dim_t DimM>
    static void apply_directional_stiffness(
        const muGrid::TypedNFieldBase<Real> & delta_strain,
        const muGrid::TypedNFieldBase<Real> & tangent,
        muGrid::TypedNFieldBase<Real> & delta_stress);

    /**
     * statically dimensioned worker for evaluating the incremental tangent
     * operator
     */
    template <Dim_t DimM>
    static void add_projected_directional_stiffness_helper(
        const muGrid::TypedNFieldBase<Real> & delta_strain,
        const muGrid::TypedNFieldBase<Real> & tangent, const Real & alpha,
        muGrid::TypedNFieldBase<Real> & delta_stress);

    //! helper function for the globalise_<T>_internal_field() functions
    template <typename T>
    muGrid::TypedNField<T> &
    globalise_internal_field(const std::string & unique_name);
    bool initialised{false};  //!< to handle double initialisations right
    //! container of the materials present in the cell
    std::vector<Material_ptr> materials{};

    Projection_ptr projection;  //!< handle for the projection operator

    //! handle for the global fields associated with this cell
    std::unique_ptr<muGrid::GlobalNFieldCollection> fields;
    muGrid::RealNField & strain;  //!< ref to strain field
    muGrid::RealNField & stress;  //!< ref to stress field
    //! Tangent field might not even be required; so this is an
    //! optional ref_wrapper instead of a ref
    optional<std::reference_wrapper<muGrid::RealNField>> tangent{};
  };

}  // namespace muSpectre

#endif  // SRC_CELL_NCELL_HH_