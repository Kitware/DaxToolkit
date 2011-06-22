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
        float functions[8];
        float rm, sm, tm;
        float r, s, t;
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
};

#endif
