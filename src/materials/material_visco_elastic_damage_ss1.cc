/**
 * @file material_visco_elastic_damage_ss1.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   20 Dec 2019
 *
 * @brief  The implementation of the methods of the
 * MaterialViscoElasticDamageSS1
 *
 * Copyright © 2019 Ali Falsafi
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

#include "materials/material_visco_elastic_damage_ss1.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialViscoElasticDamageSS1<DimM>::MaterialViscoElasticDamageSS1(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts, const Real & young_inf, const Real & young_v,
      const Real & eta_v, const Real & poisson, const Real & kappa_init,
      const Real & alpha, const Real & beta, const Real & dt,
      const std::shared_ptr<muGrid::LocalFieldCollection> &
          parent_field_collection)
      : Parent{name, spatial_dimension, nb_quad_pts, parent_field_collection},
        material_child(name + "_child", spatial_dimension, nb_quad_pts,
                       young_inf, young_v, eta_v, poisson, dt,
                       this->internal_fields),
        kappa_field{this->get_prefix() + "strain measure",
                    *this->internal_fields, QuadPtTag},
        kappa_init{kappa_init}, alpha{alpha}, beta{beta} {}

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialViscoElasticDamageSS1<DimM>::save_history_variables() {
    this->get_history_integral().get_state_field().cycle();
    this->get_s_null_prev_field().get_state_field().cycle();
    this->get_kappa_field().get_state_field().cycle();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialViscoElasticDamageSS1<DimM>::initialise() {
    Parent::initialise();
    this->get_history_integral().get_map().get_current() =
        Eigen::Matrix<Real, DimM, DimM>::Identity();
    this->get_s_null_prev_field().get_map().get_current() =
        Eigen::Matrix<Real, DimM, DimM>::Identity();
    this->kappa_field.get_map().get_current() = this->kappa_init;
    this->save_history_variables();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialViscoElasticDamageSS1<DimM>::evaluate_stress(
      const Eigen::Ref<const T2_t> & E, T2StRef_t h_prev, T2StRef_t s_null_prev,
      ScalarStRef_t kappa) -> T2_t {
    this->update_damage_measure(E, kappa);
    auto && damage{this->compute_damage_measure(kappa.current())};
    T2_t S{damage *
           this->material_child.evaluate_stress(E, h_prev, s_null_prev)};
    return S;
  }

  /* ----------------------------------------------------------------------*/
  template <Index_t DimM>
  auto MaterialViscoElasticDamageSS1<DimM>::evaluate_stress_tangent(
      const Eigen::Ref<const T2_t> & E, T2StRef_t h_prev, T2StRef_t s_null_prev,
      ScalarStRef_t kappa) -> std::tuple<T2_t, T4_t> {
    this->update_damage_measure(E, kappa);
    auto && damage{this->compute_damage_measure(kappa.current())};
    auto && SC_pristine{
        this->material_child.evaluate_stress_tangent(E, h_prev, s_null_prev)};
    auto && S{damage * std::get<0>(SC_pristine)};
    auto && C{damage * std::get<1>(SC_pristine)};
    return std::make_tuple(S, C);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialViscoElasticDamageSS1<DimM>::update_damage_measure(
      const Eigen::Ref<const T2_t> & E, ScalarStRef_t kappa) {
    auto && kappa_current{this->compute_strain_measure(E)};
    auto && kappa_old{kappa.old()};

    if (kappa_current > kappa_old) {
      kappa.current() = kappa_current;
    } else {
      kappa.current() = kappa.old();
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  Real MaterialViscoElasticDamageSS1<DimM>::compute_strain_measure(
      const Eigen::MatrixBase<Derived> & E) {
    auto && elastic_stress{this->material_child.evaluate_elastic_stress(E)};
    // Different strain measures can be considered here.
    // In this material, the considered strain measure is a strain energy
    // measure = √(Sₑₗₐ : E) →
    return sqrt(muGrid::Matrices::ddot<DimM>(elastic_stress, E));
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  Real MaterialViscoElasticDamageSS1<DimM>::compute_damage_measure(
      const Real & kappa) {
    auto && damage_measure{
        this->beta +
        (1.0 - this->beta) *
            ((1.0 - std::exp(-(kappa - this->kappa_init) / this->alpha)) /
             ((kappa - this->kappa_init) / this->alpha))};
    if (std::isnan(damage_measure) or damage_measure < 0.0) {
      return 1.0;
    } else {
      return damage_measure;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  muGrid::MappedT2StateField<Real, Mapping::Mut, DimM, IterUnit::SubPt> &
  MaterialViscoElasticDamageSS1<DimM>::get_history_integral() {
    return this->material_child.get_history_integral();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  muGrid::MappedT2StateField<Real, Mapping::Mut, DimM, IterUnit::SubPt> &
  MaterialViscoElasticDamageSS1<DimM>::get_s_null_prev_field() {
    return this->material_child.get_s_null_prev_field();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt> &
  MaterialViscoElasticDamageSS1<DimM>::get_kappa_field() {
    return this->kappa_field;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  const Real & MaterialViscoElasticDamageSS1<DimM>::get_kappa_init() {
    return this->kappa_init;
  }

  /* ----------------------------------------------------------------------*/
  template class MaterialViscoElasticDamageSS1<twoD>;
  template class MaterialViscoElasticDamageSS1<threeD>;

}  // namespace muSpectre
