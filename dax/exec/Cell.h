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
  const dax::StructuredPointsMetaData &GridStructure;
  const dax::Id CellId;

public:
  /// Create a cell for the given work.
  __device__ CellVoxel(const dax::StructuredPointsMetaData &gs, dax::Id id)
    : GridStructure(gs), CellId(id) { }

  /// Get the number of points in the cell.
  __device__ dax::Id GetNumberOfPoints() const
  {
    return 8;
  }

  /// Get the work corresponding to a given point.
  __device__ dax::exec::WorkMapField GetPoint(const dax::Id index) const
  {
    dax::Int3 dims;
    dims.x = this->GridStructure.ExtentMax.x - this->GridStructure.ExtentMin.x + 1;
    dims.y = this->GridStructure.ExtentMax.y - this->GridStructure.ExtentMin.y + 1;
    dims.z = this->GridStructure.ExtentMax.z - this->GridStructure.ExtentMin.z + 1;

    dax::Int3 cell_ijk;
    cell_ijk.x = this->CellId % (dims.x - 1);
    cell_ijk.y = (this->CellId / (dims.x - 1)) % (dims.y -1 );
    cell_ijk.z = (this->CellId / ((dims.x - 1) * (dims.y -1 )));

    const dax::Int3 cellToPointIndex[8] = {
      { 0, 0, 0 },
      { 0, 0, 1 },
      { 0, 1, 1 },
      { 0, 1, 0 },
      { 1, 0, 0 },
      { 1, 0, 1 },
      { 1, 1, 1 },
      { 1, 1, 0 }
    };

    dax::Int3 point_ijk = cell_ijk + cellToPointIndex[index];

    dax::exec::WorkMapField workPoint;
    workPoint.SetItem(point_ijk.x + dims.x*(point_ijk.y + dims.y*point_ijk.z));
    return workPoint;
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
