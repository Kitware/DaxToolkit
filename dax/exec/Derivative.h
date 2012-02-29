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
  const dax::Id NUM_POINTS  = dax::exec::CellVoxel::NUM_POINTS;

  dax::Vector3 derivativeWeights[NUM_POINTS];
  dax::exec::internal::derivativeWeightsVoxel(pcoords, derivativeWeights);

  dax::Vector3 sum = dax::make_Vector3(0.0, 0.0, 0.0);
  dax::Scalar fieldValues[NUM_POINTS];
  work.GetFieldValues(point_scalar,fieldValues);

  for (dax::Id vertexId = 0; vertexId < NUM_POINTS; vertexId++)
    {
    sum = sum + fieldValues[vertexId] * derivativeWeights[vertexId];
    }

  return sum/cell.GetSpacing();
}

//-----------------------------------------------------------------------------
template<class WorkType>
DAX_EXEC_EXPORT dax::Vector3 cellDerivative(
    const WorkType &work,
    const dax::exec::CellHexahedron &,
    const dax::Vector3 &pcoords,
    const dax::exec::FieldCoordinates &fcoords,
    const dax::exec::FieldPoint<dax::Scalar> &point_scalar)
{
  //for know we are considering that a cell hexahedron
  //is actually a voxel in an unstructured grid.
  //ToDo: use a proper derivative calculation.
  const dax::Id NUM_POINTS  = dax::exec::CellVoxel::NUM_POINTS;

  dax::Vector3 derivativeWeights[NUM_POINTS];
  dax::exec::internal::derivativeWeightsVoxel(pcoords, derivativeWeights);


  dax::Vector3 spacing;
    {
    dax::Vector3 x0 = work.GetFieldValue(fcoords,0);
    dax::Vector3 x1 = work.GetFieldValue(fcoords,1);
    dax::Vector3 x2 = work.GetFieldValue(fcoords,2);
    dax::Vector3 x4 = work.GetFieldValue(fcoords,4);
    spacing = make_Vector3(x1[0] - x0[0],
                           x2[1] - x0[1],
                           x4[2] - x0[2]);
    }

  dax::Vector3 sum = dax::make_Vector3(0.0, 0.0, 0.0);
  dax::Scalar fieldValues[NUM_POINTS];
  work.GetFieldValues(point_scalar,fieldValues);

  for (dax::Id vertexId = 0; vertexId < NUM_POINTS; vertexId++)
    {
    sum = sum + fieldValues[vertexId] * derivativeWeights[vertexId];
    }


  return sum/spacing;
}

}};

#endif //__dax_exec_Derivative_h
