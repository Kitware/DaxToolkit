/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __DaxCell_h
#define __DaxCell_h

#include "DaxCommon.h"
#include "DaxCellTypes.h"
#include "DaxWork.cu"
#include "DaxField.cu"

class DaxFieldPoint;

/// Defines a cell.
/// A cell provides API to get information about the connectivity for a cell
/// i.e. the number of points that form the cell and the point ids for those
/// points.
class DaxCell
{
  const DaxWorkMapCell& Work;
public:
  /// Create a cell for the given work.
  __device__ DaxCell(const DaxWorkMapCell& work) : Work(work)
    {
    }

  /// Returns the cell type.
  __device__ DaxCellType GetCellType() const
    {
    return DaxArrayConnectivityTraits::GetElementsType(
      this->Work.GetCellArray());
    }

  /// Get the number of points in the cell.
  __device__ DaxId GetNumberOfPoints() const
    {
    return DaxArrayConnectivityTraits::GetNumberOfConnectedElements(
      this->Work, this->Work.GetCellArray());
    }

  /// Get the work corresponding to a given point.
  __device__ DaxWorkMapField GetPoint(const DaxId index) const
    {
    return DaxArrayConnectivityTraits::GetConnectedElement(
        this->Work, this->Work.GetCellArray(), index);
    }

  /// Convenience method to a get a point coordinate
  __device__ DaxVector3 GetPoint(const DaxId& index,
    const DaxFieldCoordinates& points) const
    {
    DaxWorkMapField point_work = this->GetPoint(index);
    return points.GetVector3(point_work);
    }

  /// NOTE: virtual functions are only supported in compute 2.*+

  /// Interpolate a point scalar.
  __device__ DaxScalar Interpolate(
    const DaxVector3& pcoords,
    const DaxFieldPoint& point_scalar,
    const DaxId& component_number) const
    {
    DaxScalar output = 0;
    switch (this->GetCellType())
      {
    case VOXEL:
        {
        DaxScalar functions[8];
        DaxCellVoxel::InterpolationFunctions(pcoords, functions);
        for (DaxId cc=0; cc < 8; cc++)
          {
          DaxWorkMapField point_work = this->GetPoint(cc);
          DaxScalar cur_value = point_scalar.GetScalar(point_work
            /*, component_number*/);
          output += functions[cc] * cur_value;
          }
        }
      break;
      }
    return output;
    }

  /// Compute cell derivate for a point scalar.
  __device__ DaxVector3 Derivative(
    const DaxVector3& pcoords,
    const DaxFieldCoordinates& points,
    const DaxFieldPoint& point_scalar,
    const DaxId& component_number) const
    {
    DaxVector3 output;
    switch (this->GetCellType())
      {
    case VOXEL:
        {
        DaxScalar functionDerivs[24];
        DaxVector3 x0, x1, x2, x4, spacing;
        // get derivatives in r-s-t directions
        DaxCellVoxel::InterpolationDerivs(pcoords, functionDerivs);

        x0 = this->GetPoint(0, points);
        x1  = this->GetPoint(1, points);
        spacing.x = x1.x - x0.x;

        x2 = this->GetPoint(2, points);
        spacing.y = x2.y - x0.y;

        x4 = this->GetPoint(4, points);
        spacing.z = x4.z - x0.z;

        DaxScalar values[8];
        for (DaxId cc=0; cc < 8 ; cc++)
          {
          DaxWorkMapField point_work = this->GetPoint(cc);
          values[cc] = point_scalar.GetScalar(point_work
            /*,component_number*/);
          }

        // since the x-y-z axes are aligned with r-s-t axes, only need to scale
        // the derivative values by the data spacing.
        DaxScalar derivs[3];
        for (DaxId j=0; j < 3; j++) //loop over derivative directions
          {
          DaxScalar sum = 0.0;
          for (DaxId i=0; i < 8; i++) //loop over interp. function derivatives
            {
            sum += functionDerivs[8*j + i] * values[i];
            }
          derivs[j] = sum;
          }
        output = make_DaxVector3(derivs[0]/spacing.x, derivs[1]/spacing.y,
          derivs[2]/spacing.z);
        }
      break;

    default:
      output = make_DaxVector3(0, 0, 0);
      }
    return output;
    }
};

#endif
