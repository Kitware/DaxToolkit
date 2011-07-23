/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __DaxDataArrayStructuredConnectivity_h
#define __DaxDataArrayStructuredConnectivity_h

#include "DaxDataArrayStructuredPoints.h"
#include "DaxCellTypes.h"

/// DaxDataArrayStructuredConnectivity is used for cell-array (vtkCellArray) for a
/// structured dataset.
class DaxDataArrayStructuredConnectivity : public DaxDataArrayStructuredPoints
{
protected:
  friend class DaxDataArrayConnectivityTraits;

  __device__ static DaxId GetNumberOfConnectedElements(
    const DaxWork&, const DaxDataArray&)
    {
    return 8;
    }

  __device__ static DaxWorkMapField GetConnectedElement(
    const DaxWork& work, const DaxDataArray& connectivityArray,
    DaxId index)
    {
    DaxStructuredPointsMetaData* metadata =
      reinterpret_cast<DaxStructuredPointsMetaData*>(
      connectivityArray.RawData);

    DaxId flat_id = work.GetItem();
    // given the flat_id, what is the ijk value?
    int3 dims;
    dims.x = metadata->ExtentMax.x - metadata->ExtentMin.x + 1;
    dims.y = metadata->ExtentMax.y - metadata->ExtentMin.y + 1;
    dims.z = metadata->ExtentMax.z - metadata->ExtentMin.z + 1;

    int3 cell_ijk;
    cell_ijk.x = flat_id % (dims.x -1);
    cell_ijk.y = (flat_id / (dims.x - 1)) % (dims.y -1);
    cell_ijk.z = (flat_id / ((dims.x-1) * (dims.y -1)));

    int3 point_ijk;
    point_ijk.x = cell_ijk.x + (index % 2);
    point_ijk.y = cell_ijk.y + ((index % 4) / 2);
    point_ijk.z = cell_ijk.z + (index / 4);

    DaxWorkMapField workPoint;
    workPoint.SetItem(
      point_ijk.x + point_ijk.y * dims.x + point_ijk.z * dims.x * dims.y);
    return workPoint;
    }

  __device__ static DaxCellType GetElementsType(const DaxDataArray& connectivityArray)
    {
    DaxStructuredPointsMetaData* metadata =
      reinterpret_cast<DaxStructuredPointsMetaData*>(connectivityArray.RawData);
    int3 dims;
    dims.x = metadata->ExtentMax.x - metadata->ExtentMin.x + 1;
    dims.y = metadata->ExtentMax.y - metadata->ExtentMin.y + 1;
    dims.z = metadata->ExtentMax.z - metadata->ExtentMin.z + 1;
    int count = 0;
    count += (dims.x > 0)? 1 : 0;
    count += (dims.y > 0)? 1 : 0;
    count += (dims.z > 0)? 1 : 0;
    if (dims.x < 1 && dims.y < 1 && dims.z < 1)
      {
      return EMPTY_CELL;
      }
    else if (count == 3)
      {
      return VOXEL;
      }
    else if (count == 2)
      {
      return QUAD;
      }
    else if (count == 1)
      {
      return LINE;
      }
    return EMPTY_CELL;
    }
};

#endif
