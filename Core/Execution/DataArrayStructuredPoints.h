/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_core_exec_DataArrayStructuredPoints_h
#define __dax_core_exec_DataArrayStructuredPoints_h

#include "Core/Common/DataArray.h"
#include "Core/Execution/Work.h"

namespace dax { namespace core { namespace exec {

/// DataArrayStructuredPoints is used for point-coordinates for an uniform grid
/// (vtkImageData).
class DataArrayStructuredPoints : public dax::core::DataArray
{
protected:
  friend class dax::core::exec::DataArrayGetterTraits;

  __device__ static dax::Vector3 GetVector3(
    const dax::core::exec::Work& work, const dax::core::DataArray& array)
    {
    dax::StructuredPointsMetaData* metadata =
      reinterpret_cast<dax::StructuredPointsMetaData*>(array.RawData);

    dax::Id flat_id = work.GetItem();

    // given the flat_id, what is the ijk value?
    dax::Int3 dims;
    dims.x = metadata->ExtentMax.x - metadata->ExtentMin.x + 1;
    dims.y = metadata->ExtentMax.y - metadata->ExtentMin.y + 1;
    dims.z = metadata->ExtentMax.z - metadata->ExtentMin.z + 1;

    dax::Int3 point_ijk;
    point_ijk.x = flat_id % dims.x;
    point_ijk.y = (flat_id / dims.x)  % dims.y;
    point_ijk.z = flat_id / (dims.x * dims.y);

    dax::Vector3 point;
    point.x = metadata->Origin.x + (point_ijk.x + metadata->ExtentMin.x) * metadata->Spacing.x;
    point.y = metadata->Origin.y + (point_ijk.y + metadata->ExtentMin.y) * metadata->Spacing.y;
    point.z = metadata->Origin.z + (point_ijk.z + metadata->ExtentMin.z) * metadata->Spacing.z;
    return point;
    }
};

}}}

#endif
