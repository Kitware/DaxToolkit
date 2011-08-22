/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/// Defines various access traits for dax::core::DataArray
#ifndef __dax_core_exec_DataArrayTraits_h
#define __dax_core_exec_DataArrayTraits_h

#include "BasicTypes/Common/Types.h"
#include "Core/Common/CellTypes.h"
#include "Core/Execution/Work.h"
#include "Core/Execution/DataArrayIrregular.h"
#include "Core/Execution/DataArrayStructuredConnectivity.h"
#include "Core/Execution/DataArrayStructuredPoints.h"

namespace dax { namespace core { namespace exec {

class DataArrayConnectivityTraits
{
public:
  __device__ static dax::Id GetNumberOfConnectedElements(
    const dax::core::exec::Work& work, const dax::core::DataArray& array)
    {
    switch (array.Type)
      {
    case dax::core::DataArray::STRUCTURED_CONNECTIVITY:
      return dax::core::exec::DataArrayStructuredConnectivity::GetNumberOfConnectedElements(
        work, array);
    default:
      break;
      }
    return -1;
    }

  __device__ static dax::core::exec::WorkMapField GetConnectedElement(
    const dax::core::exec::Work& work, const dax::core::DataArray& connectivityArray,
    dax::Id index)
    {
    switch (connectivityArray.Type)
      {
    case dax::core::DataArray::STRUCTURED_CONNECTIVITY:
      return dax::core::exec::DataArrayStructuredConnectivity::GetConnectedElement(
        work, connectivityArray, index);
    default:
      break;
      }
    return dax::core::exec::WorkMapField();
    }

  __device__ static dax::core::CellType GetElementsType(
    const dax::core::DataArray& connectivityArray)
    {
    switch (connectivityArray.Type)
      {
    case dax::core::DataArray::STRUCTURED_CONNECTIVITY:
      return dax::core::exec::DataArrayStructuredConnectivity::GetElementsType(
        connectivityArray);
    default:
      break;
      }

    return dax::core::EMPTY_CELL;
    }
};

class DataArraySetterTraits
{
public:
  __device__ static void Set(
    const dax::core::exec::Work& work, dax::core::DataArray& array, dax::Scalar scalar)
    {
    switch (array.Type)
      {
    case dax::core::DataArray::IRREGULAR:
      return dax::core::exec::DataArrayIrregular::Set(work, array, scalar);
    default:
      break;
      }
    }

  __device__ static void Set(
    const dax::core::exec::Work& work, dax::core::DataArray& array, dax::Vector3 vector3)
    {
    switch (array.Type)
      {
    case dax::core::DataArray::IRREGULAR:
      return dax::core::exec::DataArrayIrregular::Set(work, array, vector3);

    default:
      break;
      }
    }

  __device__ static void Set(
    const dax::core::exec::Work& work, dax::core::DataArray& array, dax::Vector4 vector4)
    {
    switch (array.Type)
      {
    case dax::core::DataArray::IRREGULAR:
      return dax::core::exec::DataArrayIrregular::Set(work, array, vector4);

    default:
      break;
      }
    }
};

class DataArrayGetterTraits
{
public:
  __device__ static dax::Scalar GetScalar(const dax::core::exec::Work& work, const dax::core::DataArray& array)
    {
    switch (array.Type)
      {
    case dax::core::DataArray::IRREGULAR:
      return dax::core::exec::DataArrayIrregular::GetScalar(work, array);

    default:
      break;
      }
    return -1;
    }

  __device__ static dax::Vector3 GetVector3(const dax::core::exec::Work& work, const dax::core::DataArray& array)
    {
    switch (array.Type)
      {
    case dax::core::DataArray::IRREGULAR:
      return dax::core::exec::DataArrayIrregular::GetVector3(work, array);

    case dax::core::DataArray::STRUCTURED_POINTS:
      return dax::core::exec::DataArrayStructuredPoints::GetVector3(work, array);

    default:
      break;
      }
    return dax::make_Vector3(0, 0, 0);
    }

  __device__ static dax::Vector4 GetVector4(const dax::core::exec::Work& work, const dax::core::DataArray& array)
    {
    switch (array.Type)
      {
    case dax::core::DataArray::IRREGULAR:
      return dax::core::exec::DataArrayIrregular::GetVector4(work, array);

    default:
      break;
      }
    return dax::make_Vector4(0, 0, 0, 0);
    }
};

}}}
#endif
