/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

#ifndef __dax_exec_Derivative_h
#define __dax_exec_Derivative_h

#include <dax/exec/Cell.h>

#include <dax/internal/CellTypes.h>

namespace dax { namespace exec {

template<class CellType>
__device__ dax::Vector3 cellDerivative(
    const CellType &cell,
    const dax::Vector3 &pcoords,
    const dax::exec::FieldCoordinates &points,
    const dax::exec::FieldPoint &point_scalar,
    const dax::Id &component_number);


//-----------------------------------------------------------------------------
template<>
__device__ dax::Vector3 cellDerivative(
    const dax::exec::CellVoxel &cell,
    const dax::Vector3 &pcoords,
    const dax::exec::FieldCoordinates &points,
    const dax::exec::FieldPoint &point_scalar,
    const dax::Id &component_number)
{
  dax::Scalar functionDerivs[24];
  // Get derivates in r-s-t direction.
  // This seems like a weird place for this low level function.
  dax::internal::CellVoxel::InterpolationDerivs(pcoords, functionDerivs);

  // TODO: You should be able to get the spacing directly from a CellVoxel.
  dax::Vector3 x0, x1, x2, x4, spacing;
  x0 = cell.GetPoint(0, points);
  x1 = cell.GetPoint(1, points);
  spacing.x = x1.x - x0.x;
  x2 = cell.GetPoint(2, points);
  spacing.y = x2.y - x0.y;
  x4 = cell.GetPoint(4, points);
  spacing.z = x4.z - x0.z;

  dax::Scalar values[8];
  for (dax::Id vertexId = 0; vertexId < 8; vertexId++)
    {
    dax::exec::WorkMapField point_work = cell.GetPoint(vertexId);
    values[vertexId] = point_scalar.GetScalar(point_work /*, component_number*/);
    }

  // since the x-y-z axes are aligned with r-s-t axes, only need to scale the
  // derivative values by the data spacing.
  dax::Scalar derivs[3];
  for (dax::Id direction = 0; direction < 3; direction++)
    {
    dax::Scalar sum = 0.0;
    for (dax::Id vertexId = 0; vertexId < 8; vertexId++)
      {
      sum += functionDerivs[8*direction + vertexId] * values[vertexId];
      }
    derivs[direction] = sum;
    }
  dax::Vector3 result = dax::make_Vector3(derivs[0]/spacing.x,
                                          derivs[1]/spacing.y,
                                          derivs[2]/spacing.z);

  return result;
}

}};

#endif //__dax_exec_Derivative_h
