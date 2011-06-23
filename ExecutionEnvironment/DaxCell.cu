/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __DaxCell_h
#define __DaxCell_h

#include "DaxCommon.h"
#include "DaxWork.cu"
#include "DaxField.cu"

class DaxFieldPoint;

class DaxCellVoxel
{
public:
  __device__ static void InterpolationFunctions(
    const DaxVector3& pcoords, DaxScalar functions[8])
    {
    DaxScalar rm, sm, tm;
    DaxScalar r, s, t;
    r = pcoords.x; s = pcoords.y; t = pcoords.z;
    rm = 1.0 - r;
    sm = 1.0 - s;
    tm = 1.0 - t;
    functions[0] = rm * sm * tm;
    functions[1] = r * sm * tm;
    functions[2] = rm * s * tm;
    functions[3] = r * s * tm;
    functions[4] = rm * sm * t;
    functions[5] = r * sm * t;
    functions[6] = rm * s * t;
    functions[7] = r * s * t;
    }

  __device__ static void InterpolationDerivs(
    const DaxVector3& pcoords, DaxScalar derivs[24])
    {
    DaxScalar rm, sm, tm;

    rm = 1. - pcoords.x;
    sm = 1. - pcoords.y;
    tm = 1. - pcoords.z;

    // r derivatives
    derivs[0] = -sm*tm;
    derivs[1] = sm*tm;
    derivs[2] = -pcoords.y*tm;
    derivs[3] = pcoords.y*tm;
    derivs[4] = -sm*pcoords.z;
    derivs[5] = sm*pcoords.z;
    derivs[6] = -pcoords.y*pcoords.z;
    derivs[7] = pcoords.y*pcoords.z;

    // s derivatives
    derivs[8] = -rm*tm;
    derivs[9] = -pcoords.x*tm;
    derivs[10] = rm*tm;
    derivs[11] = pcoords.x*tm;
    derivs[12] = -rm*pcoords.z;
    derivs[13] = -pcoords.x*pcoords.z;
    derivs[14] = rm*pcoords.z;
    derivs[15] = pcoords.x*pcoords.z;

    // t derivatives
    derivs[16] = -rm*sm;
    derivs[17] = -pcoords.x*sm;
    derivs[18] = -rm*pcoords.y;
    derivs[19] = -pcoords.x*pcoords.y;
    derivs[20] = rm*sm;
    derivs[21] = pcoords.x*sm;
    derivs[22] = rm*pcoords.y;
    derivs[23] = pcoords.x*pcoords.y;
    }

private:
  __host__ __device__ DaxCellVoxel() { }
};

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

  /// Get a point coordinate
  __device__ DaxVector3 GetPoint(const DaxId& index,
    const DaxFieldCoordinates& points) const
    {
    DaxWorkMapField point_work = this->GetPoint(index);
    return points.GetVector3(point_work);
    }

  /// NOTE: virtual functions are only supported in compute 2.*+
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
