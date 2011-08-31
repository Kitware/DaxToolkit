/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_exec_internal_DataArrayTraits_h
#define __dax_exec_internal_DataArrayTraits_h

#include <dax/Types.h>
#include <dax/internal/CellTypes.h>

#include <dax/exec/Work.h>

#include <dax/exec/internal/DataArrayIrregular.h>
#include <dax/exec/internal/DataArrayStructuredConnectivity.h>
#include <dax/exec/internal/DataArrayStructuredPoints.h>

namespace dax { namespace exec { namespace internal {

class DataArraySetterTraits
{
public:
  __device__ static void Set(
    const dax::exec::Work& work, dax::internal::DataArray& array, dax::Scalar scalar)
    {
    switch (array.Type)
      {
    case dax::internal::DataArray::IRREGULAR:
      return dax::exec::internal::DataArrayIrregular::Set(work, array, scalar);
    default:
      break;
      }
    }

  __device__ static void Set(
    const dax::exec::Work& work, dax::internal::DataArray& array, dax::Vector3 vector3)
    {
    switch (array.Type)
      {
    case dax::internal::DataArray::IRREGULAR:
      return dax::exec::internal::DataArrayIrregular::Set(work, array, vector3);

    default:
      break;
      }
    }

  __device__ static void Set(
    const dax::exec::Work& work, dax::internal::DataArray& array, dax::Vector4 vector4)
    {
    switch (array.Type)
      {
    case dax::internal::DataArray::IRREGULAR:
      return dax::exec::internal::DataArrayIrregular::Set(work, array, vector4);

    default:
      break;
      }
    }
};

class DataArrayGetterTraits
{
public:
  __device__ static dax::Scalar GetScalar(const dax::exec::Work& work, const dax::internal::DataArray& array)
    {
    switch (array.Type)
      {
    case dax::internal::DataArray::IRREGULAR:
      return dax::exec::internal::DataArrayIrregular::GetScalar(work, array);

    default:
      break;
      }
    return -1;
    }

  __device__ static dax::Vector3 GetVector3(const dax::exec::Work& work, const dax::internal::DataArray& array)
    {
    switch (array.Type)
      {
    case dax::internal::DataArray::IRREGULAR:
      return dax::exec::internal::DataArrayIrregular::GetVector3(work, array);

    case dax::internal::DataArray::STRUCTURED_POINTS:
      return dax::exec::internal::DataArrayStructuredPoints::GetVector3(work, array);

    default:
      break;
      }
    return dax::make_Vector3(0, 0, 0);
    }

  __device__ static dax::Vector4 GetVector4(const dax::exec::Work& work, const dax::internal::DataArray& array)
    {
    switch (array.Type)
      {
    case dax::internal::DataArray::IRREGULAR:
      return dax::exec::internal::DataArrayIrregular::GetVector4(work, array);

    default:
      break;
      }
    return dax::make_Vector4(0, 0, 0, 0);
    }
};

}}}
#endif
