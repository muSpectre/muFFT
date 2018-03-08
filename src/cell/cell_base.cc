/**
 * @file   cell_base.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Nov 2017
 *
 * @brief  Implementation for cell base class
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

#include "cell/cell_base.hh"
#include "common/ccoord_operations.hh"
#include "common/iterators.hh"
#include "common/tensor_algebra.hh"

#include <sstream>
#include <algorithm>


namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  CellBase<DimS, DimM>::CellBase(Projection_ptr projection_)
    :resolutions{projection_->get_resolutions()},
     pixels(resolutions),
     lengths{projection_->get_lengths()},
     fields{std::make_unique<FieldCollection_t>()},
     F{make_field<StrainField_t>("Gradient", *this->fields)},
     P{make_field<StressField_t>("Piola-Kirchhoff-1", *this->fields)},
     projection{std::move(projection_)},
     form{projection->get_formulation()}
  { }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::Material_t &
  CellBase<DimS, DimM>::add_material(Material_ptr mat) {
    this->materials.push_back(std::move(mat));
    return *this->materials.back();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::FullResponse_t
  CellBase<DimS, DimM>::evaluate_stress_tangent(StrainField_t & grad) {
    if (this->initialised == false) {
      this->initialise();
    }
    //! High level compatibility checks
    if (grad.size() != this->F.size()) {
      throw std::runtime_error("Size mismatch");
    }
    constexpr bool create_tangent{true};
    this->get_tangent(create_tangent);

    for (auto & mat: this->materials) {
      mat->compute_stresses_tangent(grad, this->P, this->K.value(),
                                    this->form);
    }
    return std::tie(this->P, this->K.value());
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::StressField_t &
  CellBase<DimS, DimM>::directional_stiffness(const TangentField_t &K,
                                                const StrainField_t &delF,
                                                StressField_t &delP) {
    for (auto && tup:
           akantu::zip(K.get_map(), delF.get_map(), delP.get_map())){
      auto & k = std::get<0>(tup);
      auto & df = std::get<1>(tup);
      auto & dp = std::get<2>(tup);
      dp = Matrices::tensmult(k, df);
    }
    return this->project(delP);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::SolvVectorOut
  CellBase<DimS, DimM>::directional_stiffness_vec(const SolvVectorIn &delF) {
    if (!this->K) {
      throw std::runtime_error
        ("corrently only implemented for cases where a stiffness matrix "
         "exists");
    }
    if (delF.size() != this->nb_dof()) {
      std::stringstream err{};
      err << "input should be of size ndof = ¶(" << this->resolutions <<") × "
          << DimS << "² = "<< this->nb_dof() << " but I got " << delF.size();
      throw std::runtime_error(err.str());
    }
    const std::string out_name{"temp output for directional stiffness"};
    const std::string in_name{"temp input for directional stiffness"};

    auto & out_tempref = this->get_managed_field(out_name);
    auto & in_tempref = this->get_managed_field(in_name);
    SolvVectorOut(in_tempref.data(), this->nb_dof()) = delF;

    this->directional_stiffness(this->K.value(), in_tempref, out_tempref);
    return SolvVectorOut(out_tempref.data(), this->nb_dof());

  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  Eigen::ArrayXXd
  CellBase<DimS, DimM>::
  directional_stiffness_with_copy
    (Eigen::Ref<Eigen::ArrayXXd> delF) {
    if (!this->K) {
      throw std::runtime_error
        ("corrently only implemented for cases where a stiffness matrix "
         "exists");
    }
    const std::string out_name{"temp output for directional stiffness"};
    const std::string in_name{"temp input for directional stiffness"};

    auto & out_tempref = this->get_managed_field(out_name);
    auto & in_tempref = this->get_managed_field(in_name);
    in_tempref.eigen() = delF;
    this->directional_stiffness(this->K.value(), in_tempref, out_tempref);
    return out_tempref.eigen();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::StressField_t &
  CellBase<DimS, DimM>::project(StressField_t &field) {
    this->projection->apply_projection(field);
    return field;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::StrainField_t &
  CellBase<DimS, DimM>::get_strain() {
    if (this->initialised == false) {
      this->initialise();
    }
    return this->F;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  const typename CellBase<DimS, DimM>::StressField_t &
  CellBase<DimS, DimM>::get_stress() const {
    return this->P;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  const typename CellBase<DimS, DimM>::TangentField_t &
  CellBase<DimS, DimM>::get_tangent(bool create) {
    if (!this->K) {
      if (create) {
        this->K = make_field<TangentField_t>("Tangent Stiffness", *this->fields);
      } else {
        throw std::runtime_error
          ("K does not exist");
      }
    }
    return this->K.value();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::StrainField_t &
  CellBase<DimS, DimM>::get_managed_field(std::string unique_name) {
    if (!this->fields->check_field_exists(unique_name)) {
      return make_field<StressField_t>(unique_name, *this->fields);
    } else {
      return static_cast<StressField_t&>(this->fields->at(unique_name));
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void CellBase<DimS, DimM>::initialise(FFT_PlanFlags flags) {
    // check that all pixels have been assigned exactly one material
    this->check_material_coverage();
    // resize all global fields (strain, stress, etc)
    this->fields->initialise(this->resolutions);
    // initialise the projection and compute the fft plan
    this->projection->initialise(flags);
    this->initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void CellBase<DimS, DimM>::initialise_materials(bool stiffness) {
    for (auto && mat: this->materials) {
      mat->initialise(stiffness);
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void CellBase<DimS, DimM>::save_history_variables() {
    for (auto && mat: this->materials) {
      mat->save_history_variables();
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::iterator
  CellBase<DimS, DimM>::begin() {
    return this->pixels.begin();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename CellBase<DimS, DimM>::iterator
  CellBase<DimS, DimM>::end() {
    return this->pixels.end();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  CellAdaptor<CellBase<DimS, DimM>>
  CellBase<DimS, DimM>::get_adaptor() {
    return Adaptor(*this);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void CellBase<DimS, DimM>::check_material_coverage() {
    auto nb_pixels = CcoordOps::get_size(this->resolutions);
    std::vector<MaterialBase<DimS, DimM>*> assignments(nb_pixels, nullptr);
    for (auto & mat: this->materials) {
      for (auto & pixel: *mat) {
        auto index = CcoordOps::get_index(this->resolutions, pixel);
        auto& assignment{assignments.at(index)};
        if (assignment != nullptr) {
          std::stringstream err{};
          err << "Pixel " << pixel << "is already assigned to material '"
              << assignment->get_name()
              << "' and cannot be reassigned to material '" << mat->get_name();
          throw std::runtime_error(err.str());
        } else {
          assignments[index] = mat.get();
        }
      }
    }

    // find and identify unassigned pixels
    std::vector<Ccoord> unassigned_pixels;
    for (size_t i = 0; i < assignments.size(); ++i) {
      if (assignments[i] == nullptr) {
        unassigned_pixels.push_back(CcoordOps::get_ccoord(this->resolutions, i));
      }
    }

    if (unassigned_pixels.size() != 0) {
      std::stringstream err {};
      err << "The following pixels have were not assigned a material: ";
      for (auto & pixel: unassigned_pixels) {
        err << pixel << ", ";
      }
      err << "and that cannot be handled";
      throw std::runtime_error(err.str());
    }
  }

  template class CellBase<twoD, twoD>;
  template class CellBase<threeD, threeD>;

}  // muSpectre