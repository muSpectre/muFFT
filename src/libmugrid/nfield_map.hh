/**
 * @file   nfield_map.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   15 Aug 2019
 *
 * @brief  Implementation of the base class of all field maps
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

#ifndef SRC_LIBMUGRID_NFIELD_MAP_HH_
#define SRC_LIBMUGRID_NFIELD_MAP_HH_

#include "grid_common.hh"
#include "iterators.hh"
#include "nfield_collection.hh"

#include <type_traits>
#include <memory>
#include <functional>

namespace muGrid {
  /**
   * base class for field map-related exceptions
   */
  class NFieldMapError : public std::runtime_error {
   public:
    //! constructor
    explicit NFieldMapError(const std::string & what)
        : std::runtime_error(what) {}
    //! constructor
    explicit NFieldMapError(const char * what) : std::runtime_error(what) {}
  };

  // forward declaration
  template <typename T>
  class TypedNFieldBase;

  /**
   * Dynamically sized field map. Field maps allow iterating over the pixels or
   * quadrature points of a field and to select the shape (in a matrix sense) of
   * the iterate. For example, it allows to iterate in 2×2 matrices over the
   * quadrature points of a strain field for a two-dimensional problem.
   */
  template <typename T, Mapping Mutability>
  class NFieldMap {
   public:
    //! stored scalar type
    using Scalar = T;

    //! const-correct field depending on mapping mutability
    using NField_t =
        std::conditional_t<Mutability == Mapping::Const,
                           const TypedNFieldBase<T>, TypedNFieldBase<T>>;

    //! dynamically mapped eigen type
    using PlainType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    //! return type for iterators over this- map
    template <Mapping MutVal>
    using Return_t =
        std::conditional_t<MutVal == Mapping::Const,
                           Eigen::Map<const PlainType>, Eigen::Map<PlainType>>;

    //! Input type for matrix-like values (used for setting uniform values)
    using EigenRef = Eigen::Ref<const PlainType>;

    /**
     * zip-container for iterating over pixel index and stored value
     * simultaneously
     */
    using PixelEnumeration_t =
        akantu::containers::ZipContainer<NFieldCollection::PixelIndexIterable,
                                         NFieldMap &>;
    /**
     * zip-container for iterating over pixel or quadrature point index and
     * stored value simultaneously
     */
    using Enumeration_t =
        akantu::containers::ZipContainer<NFieldCollection::IndexIterable,
                                         NFieldMap &>;

    //! determine whether a field is mutably mapped at compile time
    constexpr static Mapping FieldMutability() { return Mutability; }

    //! determine whether a field map is statically sized at compile time
    constexpr static bool IsStatic() { return false; }

    //! Default constructor
    NFieldMap() = delete;

    /**
     * Constructor from a field. The default case is a map iterating over
     * quadrature points with a matrix of shape (nb_components × 1) per field
     * entry
     */
    explicit NFieldMap(NField_t & field,
                       Iteration iter_type = Iteration::QuadPt);

    /**
     * Constructor from a field with explicitly chosen shape of iterate. (the
     * number of columns is inferred).
     */
    NFieldMap(NField_t & field, Dim_t nb_rows,
              Iteration iter_type = Iteration::QuadPt);

    //! Copy constructor
    NFieldMap(const NFieldMap & other) = delete;

    //! Move constructor
    NFieldMap(NFieldMap && other);

    //! Destructor
    virtual ~NFieldMap() = default;

    //! Copy assignment operator (delete because of reference member)
    NFieldMap & operator=(const NFieldMap & other) = delete;

    //! Move assignment operator (delete because of reference member)
    NFieldMap & operator=(NFieldMap && other) = delete;

    //! Assign a matrix-like value to every entry
    template <bool IsMutableField = Mutability == Mapping::Mut>
    std::enable_if_t<IsMutableField, NFieldMap> &
    operator=(const EigenRef & val) {
      if (not((val.rows() == this->nb_rows) and
              (val.cols() == this->nb_cols))) {
        std::stringstream error_str{};
        error_str << "Expected an array/matrix with shape (" << this->nb_rows
                  << " × " << this->nb_cols
                  << "), but received a value of shape (" << val.rows() << " × "
                  << val.cols() << ").";
        throw NFieldMapError(error_str.str());
      }
      for (auto && entry : *this) {
        entry = val;
      }
      return *this;
    }

    //! Assign a scalar value to every entry
    template <bool IsMutableField = Mutability == Mapping::Mut>
    std::enable_if_t<IsMutableField, NFieldMap> &
    operator=(const Scalar & val) {
      if (not(this->nb_rows == 1 && this->nb_cols == 1)) {
        std::stringstream error_str{};
        error_str << "Expected an array/matrix with shape (" << this->nb_rows
                  << " × " << this->nb_cols
                  << "), but received a scalar value.";
        throw NFieldMapError(error_str.str());
      }
      for (auto && entry : *this) {
        entry(0, 0) = val;
      }
      return *this;
    }

    /**
     * forward-declaration for `mugrid::NFieldMap`'s iterator
     */
    template <Mapping MutIter>
    class Iterator;
    //! stl
    using iterator =
        Iterator<(Mutability == Mapping::Mut) ? Mapping::Mut : Mapping::Const>;

    //! stl
    using const_iterator = Iterator<Mapping::Const>;

    //! stl
    iterator begin();

    //! stl
    iterator end();

    //! stl
    const_iterator cbegin();

    //! stl
    const_iterator cend();

    //! stl
    const_iterator begin() const;

    //! stl
    const_iterator end() const;

    /**
     * returns the number of iterates produced by this map (corresponds to the
     * number of field entries if Iteration::Quadpt, or the number of
     * pixels/voxels if Iteration::Pixel);
     */
    size_t size() const;

    //! random acces operator
    Return_t<Mutability> operator[](size_t index) {
      assert(this->is_initialised);
      return Return_t<Mutability>{this->data_ptr + index * this->stride,
                                  this->nb_rows, this->nb_cols};
    }

    //! random const acces operator
    Return_t<Mapping::Const> operator[](size_t index) const {
      assert(this->is_initialised);
      return Return_t<Mapping::Const>{this->data_ptr + index * this->stride,
                                      this->nb_rows, this->nb_cols};
    }

    //! query the size from the field's collection and set data_ptr
    void set_data_ptr();

    /**
     * return an iterable proxy over pixel indices and stored values
     * simultaneously. Throws a `muGrid::NFieldMapError` if the iteration type
     * is over quadrature points
     */
    PixelEnumeration_t enumerate_pixel_indices_fast();

    /**
     * return an iterable proxy over pixel/quadrature indices and stored values
     * simultaneously
     */
    Enumeration_t enumerate_indices();

    //! evaluate and return the mean value of the map
    PlainType mean() const;

   protected:
    //! mapped field. Needed for query at initialisations
    const NField_t & field;
    const Iteration iteration;  //!< type of map iteration
    const Dim_t stride;         //!< precomputed stride
    const Dim_t nb_rows;        //!< number of rows of the iterate
    const Dim_t nb_cols;        //!< number of columns fo the iterate

    /**
     * Pointer to mapped data; is also unknown at construction and set in the
     * map's begin function
     */
    T * data_ptr{nullptr};

    //! keeps track of whether the map has been initialised.
    bool is_initialised{false};
    //! shared_ptr used for latent initialisation
    std::shared_ptr<std::function<void()>> callback{nullptr};
  };

  template <typename T, Mapping Mutability>
  template <Mapping MutIter>
  class NFieldMap<T, Mutability>::Iterator {
   public:
    //! convenience alias
    using NFieldMap_t = std::conditional_t<MutIter == Mapping::Const,
                                           const NFieldMap, NFieldMap>;
    //! stl
    using value_type =
        typename NFieldMap<T, Mutability>::template Return_t<MutIter>;

    //! stl
    using cvalue_type =
        typename NFieldMap<T, Mutability>::template Return_t<Mapping::Const>;

    //! Default constructor
    Iterator() = delete;

    //! Constructor to beginning, or to end
    Iterator(NFieldMap_t & map, bool end)
        : map{map}, index{end ? map.size() : 0} {}

    //! Copy constructor
    Iterator(const Iterator & other) = delete;

    //! Move constructor
    Iterator(Iterator && other) = default;

    //! Destructor
    virtual ~Iterator() = default;

    //! Copy assignment operator
    Iterator & operator=(const Iterator & other) = default;

    //! Move assignment operator
    Iterator & operator=(Iterator && other) = default;

    //! pre-increment
    Iterator & operator++() {
      this->index++;
      return *this;
    }
    //! dereference
    inline value_type operator*() { return this->map[this->index]; }
    //! dereference
    inline cvalue_type operator*() const { return this->map[this->index]; }
    //! equality
    inline bool operator==(const Iterator & other) const {
      return this->index == other.index;
    }
    //! inequality
    inline bool operator!=(const Iterator & other) const {
      return not(*this == other);
    }

   protected:
    //! NFieldMap being iterated over
    NFieldMap_t & map;
    //! current iteration index
    size_t index;
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_NFIELD_MAP_HH_