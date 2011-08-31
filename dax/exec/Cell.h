/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_exec_Cell_h
#define __dax_exec_Cell_h

#include <dax/exec/Field.h>
#include <dax/exec/Work.h>

namespace dax { namespace exec {

/// All cell objects are expected to have the following methods defined:
///   Cell<type>(work);
///   GetNumberOfPoints() const;
///   GetPoint(index) const;
///   GetPoint(index, field) const;

/// A cell in a regular structured grid.
class CellVoxel
{
private:
  const dax::exec::WorkMapCell &Work;

public:
  /// Create a cell for the given work.
  __device__ CellVoxel(const dax::exec::WorkMapCell &work): Work(work) { }

  /// Get the number of points in the cell.
  __device__ dax::Id GetNumberOfPoints() const
  {
    return 8;
  }

  /// Get the work corresponding to a given point.
  __device__ dax::exec::WorkMapField GetPoint(const dax::Id index) const
  {
    // This seems marginally wrong.  We know we are voxel, so we should be
    // able to find points just by indexing.  That may shake itself out,
    // however, once the connectivity "array" structure is templated.
    return dax::exec::internal::DataArrayConnectivityTraits::GetConnectedElement(
        this->Work,
        this->Work.GetCellArray(),
        index);
  }

  /// Convenience method to a get a point coordinate
  __device__ dax::Vector3 GetPoint(
      const dax::Id& index,
      const dax::exec::FieldCoordinates& points) const
  {
    dax::exec::WorkMapField point_work = this->GetPoint(index);
    return points.GetVector3(point_work);
  }
};

}}
#endif
