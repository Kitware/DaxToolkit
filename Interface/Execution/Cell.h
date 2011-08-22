/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_exec_Cell_h
#define __dax_exec_Cell_h

#include "Core/Common/CellTypes.h"
#include "Interface/Execution/Field.h"
#include "Interface/Execution/Work.h"

namespace dax { namespace exec {

/// Defines a cell.
/// A cell provides API to get information about the connectivity for a cell
/// i.e. the number of points that form the cell and the point ids for those
/// points.
class Cell
{
  const dax::exec::WorkMapCell& Work;
public:
  /// Create a cell for the given work.
  __device__ Cell(const dax::exec::WorkMapCell& work) : Work(work)
    {
    }

  /// Returns the cell type.
  __device__ dax::core::CellType GetCellType() const
    {
    return dax::core::exec::DataArrayConnectivityTraits::GetElementsType(
      this->Work.GetCellArray());
    }

  /// Get the number of points in the cell.
  __device__ dax::Id GetNumberOfPoints() const
    {
    return dax::core::exec::DataArrayConnectivityTraits::GetNumberOfConnectedElements(
      this->Work, this->Work.GetCellArray());
    }

  /// Get the work corresponding to a given point.
  __device__ dax::exec::WorkMapField GetPoint(const dax::Id index) const
    {
    return dax::core::exec::DataArrayConnectivityTraits::GetConnectedElement(
        this->Work, this->Work.GetCellArray(), index);
    }

  /// Convenience method to a get a point coordinate
  __device__ dax::Vector3 GetPoint(const dax::Id& index,
    const dax::exec::FieldCoordinates& points) const
    {
    dax::exec::WorkMapField point_work = this->GetPoint(index);
    return points.GetVector3(point_work);
    }

  /// NOTE: virtual functions are only supported in compute 2.*+

  /// Interpolate a point scalar.
  __device__ dax::Scalar Interpolate(
    const dax::Vector3& pcoords,
    const dax::exec::FieldPoint& point_scalar,
    const dax::Id& component_number) const
    {
    dax::Scalar output = 0;
    switch (this->GetCellType())
      {
    case dax::core::VOXEL:
        {
        dax::Scalar functions[8];
        dax::core::CellVoxel::InterpolationFunctions(pcoords, functions);
        for (dax::Id cc=0; cc < 8; cc++)
          {
          dax::exec::WorkMapField point_work = this->GetPoint(cc);
          dax::Scalar cur_value = point_scalar.GetScalar(point_work
            /*, component_number*/);
          output += functions[cc] * cur_value;
          }
        }
      break;

    default:
      break;
      }
    return output;
    }

  /// Compute cell derivate for a point scalar.
  __device__ dax::Vector3 Derivative(
    const dax::Vector3& pcoords,
    const dax::exec::FieldCoordinates& points,
    const dax::exec::FieldPoint& point_scalar,
    const dax::Id& component_number) const
    {
    dax::Vector3 output;
    switch (this->GetCellType())
      {
    case dax::core::VOXEL:
        {
        dax::Scalar functionDerivs[24];
        dax::Vector3 x0, x1, x2, x4, spacing;
        // get derivatives in r-s-t directions
        dax::core::CellVoxel::InterpolationDerivs(pcoords, functionDerivs);

        x0 = this->GetPoint(0, points);
        x1  = this->GetPoint(1, points);
        spacing.x = x1.x - x0.x;

        x2 = this->GetPoint(2, points);
        spacing.y = x2.y - x0.y;

        x4 = this->GetPoint(4, points);
        spacing.z = x4.z - x0.z;

        dax::Scalar values[8];
        for (dax::Id cc=0; cc < 8 ; cc++)
          {
          dax::exec::WorkMapField point_work = this->GetPoint(cc);
          values[cc] = point_scalar.GetScalar(point_work /*,component_number*/);
          }

        // since the x-y-z axes are aligned with r-s-t axes, only need to scale
        // the derivative values by the data spacing.
        dax::Scalar derivs[3];
        for (dax::Id j=0; j < 3; j++) //loop over derivative directions
          {
          dax::Scalar sum = 0.0;
          for (dax::Id i=0; i < 8; i++) //loop over interp. function derivatives
            {
            sum += functionDerivs[8*j + i] * values[i];
            }
          derivs[j] = sum;
          }
        output = dax::make_Vector3(derivs[0]/spacing.x, derivs[1]/spacing.y,
          derivs[2]/spacing.z);
        }
      break;

    default:
      output = dax::make_Vector3(0, 0, 0);
      }
    return output;
    }
};

}}
#endif
