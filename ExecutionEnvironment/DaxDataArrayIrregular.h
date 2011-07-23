/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __DaxDataArrayIrregular_h
#define __DaxDataArrayIrregular_h

#include "DaxDataArray.h"
#include "DaxWork.h"

/// DaxDataArrayIrregular represents a C/C++ array of values. It is typically used
/// attributes that have no structure.
class DaxDataArrayIrregular : public DaxDataArray
{
protected:
  //---------------------------------------------------------------------------
  friend class DaxDataArraySetterTraits;

  __device__ static void Set(const DaxWork& work, DaxDataArray& array, DaxScalar scalar)
    {
    int num_comps = DaxDataArrayIrregular::GetNumberOfComponents(array);
    int index = 0; // FIXME: add API for this
    reinterpret_cast<DaxScalar*>(array.RawData)[work.GetItem() * num_comps + index] = scalar;
    }

  __device__ static void Set(const DaxWork& work, DaxDataArray& array, DaxVector3 value)
    {
    int num_comps = DaxDataArrayIrregular::GetNumberOfComponents(array);
    DaxScalar* ptr = &reinterpret_cast<DaxScalar*>(array.RawData)[work.GetItem() * num_comps];
    ptr[0] = value.x;
    ptr[1] = value.y;
    ptr[2] = value.z;
    }

  __device__ static void Set(const DaxWork& work, DaxDataArray& array, DaxVector4 value)
    {
    int num_comps = DaxDataArrayIrregular::GetNumberOfComponents(array);
    DaxScalar* ptr = &reinterpret_cast<DaxScalar*>(array.RawData)[work.GetItem() * num_comps];
    ptr[0] = value.x;
    ptr[1] = value.y;
    ptr[2] = value.z;
    ptr[3] = value.w;
    }

  //---------------------------------------------------------------------------
  friend class DaxDataArrayGetterTraits;

  __device__ static DaxScalar GetScalar(const DaxWork& work, const DaxDataArray& array)
    {
    int num_comps = DaxDataArrayIrregular::GetNumberOfComponents(array);
    int index = 0; // FIXME: add API for this
    return reinterpret_cast<DaxScalar*>(array.RawData)[work.GetItem() * num_comps + index];
    }

  __device__ static DaxVector3 GetVector3(const DaxWork& work, const DaxDataArray& array)
    {
    int num_comps = DaxDataArrayIrregular::GetNumberOfComponents(array);
    DaxScalar* ptr = &reinterpret_cast<DaxScalar*>(array.RawData)[work.GetItem() * num_comps];
    return make_DaxVector3(ptr[0], ptr[1], ptr[2]);
    }

  __device__ static DaxVector4 GetVector4(const DaxWork& work, const DaxDataArray& array)
    {
    int num_comps = DaxDataArrayIrregular::GetNumberOfComponents(array);
    DaxScalar* ptr = &reinterpret_cast<DaxScalar*>(array.RawData)[work.GetItem() * num_comps];
    return make_DaxVector4(ptr[0], ptr[1], ptr[2], ptr[3]);
    }
};

#endif
