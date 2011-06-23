/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __DaxArrayIrregular_h
#define __DaxArrayIrregular_h

#include "DaxArray.cu"

/// DaxArrayIrregular represents a C/C++ array of values. It is typically used
/// attributes that have no structure.
class DaxArrayIrregular : public DaxArray
{
  SUPERCLASS(DaxArray);
  DaxId NumberOfTuples;
  DaxId NumberOfComponents;
public:
  __host__ DaxArrayIrregular() :
    NumberOfTuples(0),
    NumberOfComponents(0)
    {
    this->Type = IRREGULAR;
    }

  __host__ void SetNumberOfTuples(DaxId val)
    {
    this->NumberOfTuples = val;
    }

  __host__ void SetNumberOfComponents(DaxId val)
    {
    this->NumberOfComponents = val;
    switch (val)
      {
    case 1:
      this->DataType = SCALAR;
      break;
    case 3:
      this->DataType = VECTOR3;
      break;
    case 4:
      this->DataType = VECTOR4;
      break;
    default:
      abort();
      }
    }

  __host__ void Allocate()
    {
    this->Superclass::Allocate(
      this->NumberOfComponents * this->NumberOfTuples * sizeof(float));
    }

  __host__ void SetValue(DaxId tupleId, DaxId componentId, float value)
    {
    reinterpret_cast<float*>(this->RawData)
      [tupleId * this->NumberOfComponents + componentId] = value;
    }
  __host__ float GetValue(DaxId tupleId, DaxId componentId)
    {
    return reinterpret_cast<float*>(this->RawData)
      [tupleId * this->NumberOfComponents + componentId];
    }
protected:
  //---------------------------------------------------------------------------
  friend class DaxArraySetterTraits;

  __device__ static void Set(const DaxWork& work, DaxArray& array, DaxScalar scalar)
    {
    int num_comps = DaxArrayIrregular::GetNumberOfComponents(array);
    int index = 0; // FIXME: add API for this
    reinterpret_cast<DaxScalar*>(array.RawData)[work.GetItem() * num_comps + index] = scalar;
    }

  __device__ static void Set(const DaxWork& work, DaxArray& array, DaxVector3 value)
    {
    int num_comps = DaxArrayIrregular::GetNumberOfComponents(array);
    DaxScalar* ptr = &reinterpret_cast<DaxScalar*>(array.RawData)[work.GetItem() * num_comps];
    ptr[0] = value.x;
    ptr[1] = value.y;
    ptr[2] = value.z;
    }

  __device__ static void Set(const DaxWork& work, DaxArray& array, DaxVector4 value)
    {
    int num_comps = DaxArrayIrregular::GetNumberOfComponents(array);
    DaxScalar* ptr = &reinterpret_cast<DaxScalar*>(array.RawData)[work.GetItem() * num_comps];
    ptr[0] = value.x;
    ptr[1] = value.y;
    ptr[2] = value.z;
    ptr[3] = value.w;
    }

  //---------------------------------------------------------------------------
  friend class DaxArrayGetterTraits;

  __device__ static DaxScalar GetScalar(const DaxWork& work, const DaxArray& array)
    {
    int num_comps = DaxArrayIrregular::GetNumberOfComponents(array);
    int index = 0; // FIXME: add API for this
    return reinterpret_cast<DaxScalar*>(array.RawData)[work.GetItem() * num_comps + index];
    }

  __device__ static DaxVector3 GetVector3(const DaxWork& work, const DaxArray& array)
    {
    int num_comps = DaxArrayIrregular::GetNumberOfComponents(array);
    DaxScalar* ptr = &reinterpret_cast<DaxScalar*>(array.RawData)[work.GetItem() * num_comps];
    return make_DaxVector3(ptr[0], ptr[1], ptr[2]);
    }

  __device__ static DaxVector4 GetVector4(const DaxWork& work, const DaxArray& array)
    {
    int num_comps = DaxArrayIrregular::GetNumberOfComponents(array);
    DaxScalar* ptr = &reinterpret_cast<DaxScalar*>(array.RawData)[work.GetItem() * num_comps];
    return make_DaxVector4(ptr[0], ptr[1], ptr[2], ptr[3]);
    }
};

#endif
