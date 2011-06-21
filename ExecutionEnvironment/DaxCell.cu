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
class DaxCell
{
  const DaxWorkMapCell& Work;
public:
  /// Create a cell for the given work.
  __device__ DaxCell(const DaxWorkMapCell& work) : Work(work)
    {
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
    const DaxVector3& parametric_location,
    const DaxFieldPoint& point_scalar,
    const DaxId& component_number)
    {
    return -12;
    }
};

#endif
