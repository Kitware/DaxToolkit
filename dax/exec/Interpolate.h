/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

#ifndef __dax_exec_Interpolate_h
#define __dax_exec_Interpolate_h

#include <dax/exec/Cell.h>

#include <dax/internal/CellTypes.h>

namespace dax { namespace exec {

template<class CellType>
__device__ dax::Scalar cellInterpolate(
    const CellType &cell,
    const dax::Vector3 &pcoords,
    const dax::exec::FieldPoint &point_coords,
    const dax::Id &component_number);

//-----------------------------------------------------------------------------
template<>
__device__ dax::Scalar cellInterpolate(
    const dax::exec::CellVoxel &cell,
    const dax::Vector3 &pcoords,
    const dax::exec::FieldPoint &point_scalar,
    const dax::Id &component_number)
{
  dax::Scalar functions[8];
  // This seems like a weird place for this low level function.
  // Also, I suggest we order the vertices of voxels in CGNS order (i.e., the
  // same as a hexahedron).
  dax::internal::CellVoxel::InterpolationFunctions(pcoords, functions);

  dax::Scalar result = 0;

  for (dax::Id vertexId = 0; vertexId < 8; vertexId++)
    {
    dax::exec::WorkMapField point_work = cell.GetPoint(vertexId);
    dax::Scalar cur_value = point_scalar.GetScalar(point_work /*, component_number*/);
    result += functions[vertexId] * cur_value;
    }

  return result;
}

}};

#endif //__dax_exec_Interpolate_h
