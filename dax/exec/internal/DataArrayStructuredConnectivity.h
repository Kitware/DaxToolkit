/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_exec_internal_DataArrayStructuredConnectivity_h
#define __dax_exec_internal_DataArrayStructuredConnectivity_h

#include <dax/internal/CellTypes.h>

#include <dax/exec/internal/DataArrayStructuredPoints.h>

namespace dax { namespace exec { namespace internal {

  class DataArrayConnectivityTraits;

/// DataArrayStructuredConnectivity is used for cell-array (vtkCellArray) for a
/// structured dataset.
class DataArrayStructuredConnectivity : public dax::exec::internal::DataArrayStructuredPoints
{
protected:
  friend class dax::exec::internal::DataArrayConnectivityTraits;

  __device__ static dax::Id GetNumberOfConnectedElements(
    const dax::exec::Work&, const dax::internal::DataArray&)
    {
    return 8;
    }

  __device__ static dax::exec::WorkMapField GetConnectedElement(
    const dax::exec::Work& work, const dax::internal::DataArray& connectivityArray,
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

    dax::exec::WorkMapField workPoint;
    workPoint.SetItem(
      point_ijk.x + point_ijk.y * dims.x + point_ijk.z * dims.x * dims.y);
    return workPoint;
    }

  __device__ static dax::internal::CellType GetElementsType(const dax::internal::DataArray& connectivityArray)
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
      return dax::internal::EMPTY_CELL;
      }
    else if (count == 3)
      {
      return dax::internal::VOXEL;
      }
    else if (count == 2)
      {
      return dax::internal::QUAD;
      }
    else if (count == 1)
      {
      return dax::internal::LINE;
      }
    return dax::internal::EMPTY_CELL;
    }
};

}}}
#endif
