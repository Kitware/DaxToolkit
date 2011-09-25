/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

#ifndef __dax_exec_Derivative_h
#define __dax_exec_Derivative_h

#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>

#include <dax/exec/internal/DerivativeWeights.h>

namespace dax { namespace exec {

//-----------------------------------------------------------------------------
template<class WorkType>
DAX_EXEC_EXPORT dax::Vector3 cellDerivative(
    const WorkType &work,
    const dax::exec::CellVoxel &cell,
    const dax::Vector3 &pcoords,
    const dax::exec::FieldCoordinates &, // Not used for voxels
    const dax::exec::FieldPoint<dax::Scalar> &point_scalar)
{
  const dax::Id numVerts = 8;

  dax::Vector3 derivativeWeights[numVerts];
  dax::exec::internal::derivativeWeightsVoxel(pcoords, derivativeWeights);

  dax::Vector3 sum = dax::make_Vector3(0.0, 0.0, 0.0);
  for (dax::Id vertexId = 0; vertexId < numVerts; vertexId++)
    {
    dax::Scalar value = work.GetFieldValue(point_scalar, vertexId);
    sum = sum + value * derivativeWeights[vertexId];
    }

  return sum/cell.GetSpacing();
}

}};

#endif //__dax_exec_Derivative_h
