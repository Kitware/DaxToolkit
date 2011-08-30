/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_exec_internal_DataArrayIrregular_h
#define __dax_exec_internal_DataArrayIrregular_h

#include <dax/Types.h>
#include <dax/internal/DataArray.h>

#include <dax/exec/Work.h>

namespace dax { namespace exec { namespace internal {

  class DataArraySetterTraits;
  class DataArrayGetterTraits;

/// DataArrayIrregular represents a C/C++ array of values. It is typically used
/// attributes that have no structure.
class DataArrayIrregular : public dax::internal::DataArray
{
protected:
  //---------------------------------------------------------------------------
  friend class dax::exec::internal::DataArraySetterTraits;

  __device__ static void Set(const dax::exec::Work& work, dax::internal::DataArray& array, dax::Scalar scalar)
    {
    int num_comps = dax::internal::DataArray::GetNumberOfComponents(array);
    int index = 0; // FIXME: add API for this
    reinterpret_cast<dax::Scalar*>(array.RawData)[work.GetItem() * num_comps + index] = scalar;
    }

  __device__ static void Set(const dax::exec::Work& work, dax::internal::DataArray& array, dax::Vector3 value)
    {
    int num_comps = dax::internal::DataArray::GetNumberOfComponents(array);
    dax::Scalar* ptr = &reinterpret_cast<dax::Scalar*>(array.RawData)[work.GetItem() * num_comps];
    ptr[0] = value.x;
    ptr[1] = value.y;
    ptr[2] = value.z;
    }

  __device__ static void Set(const dax::exec::Work& work, dax::internal::DataArray& array, dax::Vector4 value)
    {
    int num_comps = dax::internal::DataArray::GetNumberOfComponents(array);
    dax::Scalar* ptr = &reinterpret_cast<dax::Scalar*>(array.RawData)[work.GetItem() * num_comps];
    ptr[0] = value.x;
    ptr[1] = value.y;
    ptr[2] = value.z;
    ptr[3] = value.w;
    }

  //---------------------------------------------------------------------------
  friend class dax::exec::internal::DataArrayGetterTraits;

  __device__ static dax::Scalar GetScalar(const dax::exec::Work& work, const dax::internal::DataArray& array)
    {
    int num_comps = dax::internal::DataArray::GetNumberOfComponents(array);
    int index = 0; // FIXME: add API for this
    return reinterpret_cast<dax::Scalar*>(array.RawData)[work.GetItem() * num_comps + index];
    }

  __device__ static dax::Vector3 GetVector3(const dax::exec::Work& work, const dax::internal::DataArray& array)
    {
    int num_comps = dax::internal::DataArray::GetNumberOfComponents(array);
    dax::Scalar* ptr = &reinterpret_cast<dax::Scalar*>(array.RawData)[work.GetItem() * num_comps];
    return make_Vector3(ptr[0], ptr[1], ptr[2]);
    }

  __device__ static dax::Vector4 GetVector4(const dax::exec::Work& work, const dax::internal::DataArray& array)
    {
    int num_comps = dax::internal::DataArray::GetNumberOfComponents(array);
    dax::Scalar* ptr = &reinterpret_cast<dax::Scalar*>(array.RawData)[work.GetItem() * num_comps];
    return dax::make_Vector4(ptr[0], ptr[1], ptr[2], ptr[3]);
    }
};

}}}
#endif
