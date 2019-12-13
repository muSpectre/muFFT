/**
 * @file   nfield_collection_local.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   12 Aug 2019
 *
 * @brief  Local field collection
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

#ifndef SRC_LIBMUGRID_NFIELD_COLLECTION_LOCAL_HH_
#define SRC_LIBMUGRID_NFIELD_COLLECTION_LOCAL_HH_

#include "nfield_collection.hh"
#include "nfield_collection_global.hh"

namespace muGrid {

  /** `muGrid::LocalNFieldCollection` derives from `muGrid::NFieldCollection`
   * and stores local fields, i.e. fields that are only defined for a subset of
   * all pixels/voxels in the computational domain. The coordinates of these
   * active pixels are explicitly stored by this field collection.
   * `muGrid::LocalFieldCollection::add_pixel` allows to add individual
   * pixels/voxels to the field collection.
   */
  class LocalNFieldCollection : public NFieldCollection {
   public:
    //! alias for base class
    using Parent = NFieldCollection;
    //! Default constructor
    LocalNFieldCollection() = delete;

    /**
     * Constructor
     * @param spatial_dimension spatial dimension of the field (can be
     *                    muGrid::Unknown, e.g., in the case of the local fields
     *                    for storing internal material variables)
     * @param nb_quad_pts number of quadrature points per pixel/voxel
     */
    LocalNFieldCollection(Dim_t spatial_dimension, Dim_t nb_quad_pts);

    //! Copy constructor
    LocalNFieldCollection(const LocalNFieldCollection & other) = delete;

    //! Move constructor
    LocalNFieldCollection(LocalNFieldCollection && other) = default;

    //! Destructor
    virtual ~LocalNFieldCollection() = default;

    //! Copy assignment operator
    LocalNFieldCollection &
    operator=(const LocalNFieldCollection & other) = delete;

    //! Move assignment operator
    LocalNFieldCollection & operator=(LocalNFieldCollection && other) = delete;

    /**
     * Insert a new pixel/voxel into the collection.
     * @param global_index refers to the linear index this pixel has in the
     *                     global field collection defining the problem space
     */
    void add_pixel(const size_t & global_index);

    /**
     * Freeze the set of pixels this collection is responsible for and allocate
     * memory for all fields of the collection. NFields added lateron will have
     * their memory allocated upon construction
     */
    void initialise();

    /**
     * obtain a new field collection with the same domain and pixels
     */
    LocalNFieldCollection get_empty_clone() const;
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_NFIELD_COLLECTION_LOCAL_HH_