/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_core_exec_DataArrayStructuredConnectivity_h
#define __dax_core_exec_DataArrayStructuredConnectivity_h

#include "Core/Execution/DataArrayStructuredPoints.h"
#include "Core/Common/CellTypes.h"

namespace dax { namespace core { namespace exec {

  class DataArrayConnectivityTraits;

/// DataArrayStructuredConnectivity is used for cell-array (vtkCellArray) for a
/// structured dataset.
class DataArrayStructuredConnectivity : public dax::core::exec::DataArrayStructuredPoints
{
protected:
  friend class dax::core::exec::DataArrayConnectivityTraits;

  __device__ static dax::Id GetNumberOfConnectedElements(
    const dax::core::exec::Work&, const dax::core::DataArray&)
    {
    return 8;
    }

  __device__ static dax::core::exec::WorkMapField GetConnectedElement(
    const dax::core::exec::Work& work, const dax::core::DataArray& connectivityArray,
    dax::Id index)
    {
    dax::StructuredPointsMetaData* metadata =
      reinterpret_cast<dax::StructuredPointsMetaData*>(
      connectivityArray.RawData);

    dax::Id flat_id = work.GetItem();
    // given the flat_id, what is the ijk value?
    dax::Int3 dims;
    dims.x = metadata->ExtentMax.x - metadata->ExtentMin.x + 1;
    dims.y = metadata->ExtentMax.y - metadata->ExtentMin.y + 1;
    dims.z = metadata->ExtentMax.z - metadata->ExtentMin.z + 1;

    dax::Int3 cell_ijk;
    cell_ijk.x = flat_id % (dims.x -1);
    cell_ijk.y = (flat_id / (dims.x - 1)) % (dims.y -1);
    cell_ijk.z = (flat_id / ((dims.x-1) * (dims.y -1)));

    dax::Int3 point_ijk;
    point_ijk.x = cell_ijk.x + (index % 2);
    point_ijk.y = cell_ijk.y + ((index % 4) / 2);
    point_ijk.z = cell_ijk.z + (index / 4);

    dax::core::exec::WorkMapField workPoint;
    workPoint.SetItem(
      point_ijk.x + point_ijk.y * dims.x + point_ijk.z * dims.x * dims.y);
    return workPoint;
    }

  __device__ static dax::core::CellType GetElementsType(const dax::core::DataArray& connectivityArray)
    {
    dax::StructuredPointsMetaData* metadata =
      reinterpret_cast<dax::StructuredPointsMetaData*>(connectivityArray.RawData);
    dax::Int3 dims;
    dims.x = metadata->ExtentMax.x - metadata->ExtentMin.x + 1;
    dims.y = metadata->ExtentMax.y - metadata->ExtentMin.y + 1;
    dims.z = metadata->ExtentMax.z - metadata->ExtentMin.z + 1;
    int count = 0;
    count += (dims.x > 0)? 1 : 0;
    count += (dims.y > 0)? 1 : 0;
    count += (dims.z > 0)? 1 : 0;
    if (dims.x < 1 && dims.y < 1 && dims.z < 1)
      {
      return dax::core::EMPTY_CELL;
      }
    else if (count == 3)
      {
      return dax::core::VOXEL;
      }
    else if (count == 2)
      {
      return dax::core::QUAD;
      }
    else if (count == 1)
      {
      return LINE;
      }
    return dax::core::EMPTY_CELL;
    }
};

}}}
#endif
