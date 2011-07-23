/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __DaxDataArrayStructuredPoints_h
#define __DaxDataArrayStructuredPoints_h

#include "DaxDataArray.h"

/// DaxDataArrayStructuredPoints is used for point-coordinates for an uniform grid
/// (vtkImageData).
class DaxDataArrayStructuredPoints : public DaxDataArray
{
protected:
  friend class DaxDataArrayGetterTraits;

  __device__ static DaxVector3 GetVector3(const DaxWork& work, const DaxDataArray& array)
    {
    DaxStructuredPointsMetaData* metadata =
      reinterpret_cast<DaxStructuredPointsMetaData*>(array.RawData);

    DaxId flat_id = work.GetItem();

    // given the flat_id, what is the ijk value?
    int3 dims;
    dims.x = metadata->ExtentMax.x - metadata->ExtentMin.x + 1;
    dims.y = metadata->ExtentMax.y - metadata->ExtentMin.y + 1;
    dims.z = metadata->ExtentMax.z - metadata->ExtentMin.z + 1;

    int3 point_ijk;
    point_ijk.x = flat_id % dims.x;
    point_ijk.y = (flat_id / dims.x)  % dims.y;
    point_ijk.z = flat_id / (dims.x * dims.y);

    DaxVector3 point;
    point.x = metadata->Origin.x + (point_ijk.x + metadata->ExtentMin.x) * metadata->Spacing.x;
    point.y = metadata->Origin.y + (point_ijk.y + metadata->ExtentMin.y) * metadata->Spacing.y;
    point.z = metadata->Origin.z + (point_ijk.z + metadata->ExtentMin.z) * metadata->Spacing.z;
    return point;
    }
};

#endif
