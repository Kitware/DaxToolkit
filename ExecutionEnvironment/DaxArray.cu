/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/// This file defines the DataObject that can be used on the host as well as
/// OnDevice.
#ifndef __DaxArray_h
#define __DaxArray_h

#include "DaxCommon.h"
#include "DaxWork.cu"

#include <assert.h>

class DaxArray
{
public:
  enum eType
    {
    UNKNOWN= -1,
    IRREGULAR = 11,
    STRUCTURED_POINTS = 12,
    STRUCTURED_CONNECTIVITY=13,
    };
  enum eDataType
    {
    VOID,
    SCALAR,
    VECTOR3,
    VECTOR4,
    ID
    };

  eType Type;
  eDataType DataType;
  void* RawData;
  DaxId SizeInBytes;
  bool OnDevice;

  __device__ __host__ DaxArray()
    { 
    this->Type = UNKNOWN;
    this->RawData = NULL;
    this->OnDevice = false;
    this->SizeInBytes = 0;
    this->DataType = VOID;
    }

  __host__ void CopyFrom(const DaxArray& source)
    {
    this->Allocate(source);
    if (!this->OnDevice  && !source.OnDevice)
      {
      memcpy(this->RawData, source.RawData, source.SizeInBytes);
      }
    else if (this->OnDevice)
      {
      cudaMemcpy(this->RawData, source.RawData,
        source.SizeInBytes, cudaMemcpyHostToDevice);
      }
    else if (source.OnDevice)
      {
      cudaMemcpy(this->RawData, source.RawData,
        this->SizeInBytes, cudaMemcpyDeviceToHost);
      }

    }

  __host__ void Allocate(const DaxArray& source)
    {
    this->Type = source.Type;
    this->DataType = source.DataType;
    this->Allocate(source.SizeInBytes);
    }

  __host__ void FreeMemory()
    {
    if (this->OnDevice)
      {
      if (this->RawData && this->SizeInBytes)
        {
        cudaFree(this->RawData);
        }
      }
    else
      {
      free(this->RawData);
      }
    this->RawData = NULL;
    this->SizeInBytes = 0;
    }

protected:
  void Allocate(int size_in_bytes)
    {
    this->FreeMemory();
    if (this->OnDevice)
      {
      cudaMalloc(&this->RawData, size_in_bytes);
      }
    else
      {
      this->RawData = malloc(size_in_bytes);
      }
    this->SizeInBytes = size_in_bytes;
    }

  __device__ static int GetNumberOfComponents(const DaxArray& array)
    {
    int num_comps;
    switch (array.DataType)
      {
    case SCALAR:
      num_comps = 1;
      break;
    case VECTOR3:
      num_comps = 3;
      break;
    case VECTOR4:
      num_comps = 4;
      break;
    default:
      num_comps = 1;
      }
    return num_comps;
    }
};

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

class DaxArrayStructuredPoints : public DaxArray
{
  SUPERCLASS(DaxArray);
protected:
  struct MetadataType {
    float3 Origin;
    float3 Spacing;
    int3 ExtentMin;
    int3 ExtentMax;
  } Metadata;

public:
  __host__ DaxArrayStructuredPoints() 
    {
    this->Type = STRUCTURED_POINTS;
    this->Metadata.Origin = make_float3(0.0f, 0.0f, 0.0f);
    this->Metadata.Spacing = make_float3(1.0f, 1.0f, 1.0f);
    this->Metadata.ExtentMin = make_int3(0, 0, 0);
    this->Metadata.ExtentMax = make_int3(0, 0, 0);
    }

  __host__ void SetExtent(int xmin, int xmax,
    int ymin, int ymax, int zmin, int zmax)
    {
    this->Metadata.ExtentMin.x = xmin;
    this->Metadata.ExtentMin.y = ymin;
    this->Metadata.ExtentMin.z = zmin;

    this->Metadata.ExtentMax.x = xmax;
    this->Metadata.ExtentMax.y = ymax;
    this->Metadata.ExtentMax.z = zmax;
    }

  __host__ void SetSpacing(float x, float y, float z)
    {
    this->Metadata.Spacing.x = x;
    this->Metadata.Spacing.y = y;
    this->Metadata.Spacing.z = z;
    }

  __host__ void SetOrigin(float x, float y, float z)
    {
    this->Metadata.Origin.x = x;
    this->Metadata.Origin.y = y;
    this->Metadata.Origin.z = z;
    }

  __host__ void Allocate()
    {
    assert(this->OnDevice == false);
    this->Superclass::Allocate(sizeof(Metadata));
    memcpy(this->RawData, &this->Metadata, sizeof(Metadata)); 
    }
};

class DaxArrayStructuredConnectivity : public DaxArrayStructuredPoints
{
  SUPERCLASS(DaxArrayStructuredPoints);
public:
  __host__ DaxArrayStructuredConnectivity()
    {
    this->Type = STRUCTURED_CONNECTIVITY;
    }

protected:
  friend class DaxArrayConnectivityTraits;

  __device__ static DaxId GetNumberOfConnectedElements(
    const DaxWork&, const DaxArray&)
    {
    return 8;
    }

  __device__ static DaxWorkMapField GetConnectedElement(
    const DaxWork& work, const DaxArray& connectivityArray,
    DaxId index)
    {
    MetadataType* metadata = reinterpret_cast<MetadataType*>(
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

  __device__ static DaxCellType GetElementsType(const DaxArray& connectivityArray)
    {
    MetadataType* metadata = reinterpret_cast<MetadataType*>(connectivityArray.RawData);
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


class DaxArrayConnectivityTraits
{
public:
  __device__ static DaxId GetNumberOfConnectedElements(
    const DaxWork& work, const DaxArray& array)
    {
    switch (array.Type)
      {
    case DaxArray::STRUCTURED_CONNECTIVITY:
      return DaxArrayStructuredConnectivity::GetNumberOfConnectedElements(
        work, array);
      }
    return -1;
    }

  __device__ static DaxWorkMapField GetConnectedElement(
    const DaxWork& work, const DaxArray& connectivityArray,
    DaxId index)
    {
    switch (connectivityArray.Type)
      {
    case DaxArray::STRUCTURED_CONNECTIVITY:
      return DaxArrayStructuredConnectivity::GetConnectedElement(
        work, connectivityArray, index);
      }
    return DaxWorkMapField();
    }

  __device__ static DaxCellType GetElementsType(
    const DaxArray& connectivityArray)
    {
    switch (connectivityArray.Type)
      {
    case DaxArray::STRUCTURED_CONNECTIVITY:
      return DaxArrayStructuredConnectivity::GetElementsType(
        connectivityArray);
      }

    return EMPTY_CELL;
    }
};

class DaxArraySetterTraits
{
public:
  __device__ static void Set(
    const DaxWork& work, DaxArray& array, DaxScalar scalar)
    {
    switch (array.Type)
      {
    case DaxArray::IRREGULAR:
      return DaxArrayIrregular::Set(work, array, scalar);
      }
    }

  __device__ static void Set(
    const DaxWork& work, DaxArray& array, DaxVector3 vector3)
    {
    switch (array.Type)
      {
    case DaxArray::IRREGULAR:
      return DaxArrayIrregular::Set(work, array, vector3);
      }
    }

  __device__ static void Set(
    const DaxWork& work, DaxArray& array, DaxVector4 vector4)
    {
    switch (array.Type)
      {
    case DaxArray::IRREGULAR:
      return DaxArrayIrregular::Set(work, array, vector4);
      }
    }


};

class DaxArrayGetterTraits
{
public:
  __device__ static DaxScalar GetScalar(const DaxWork& work, const DaxArray& array)
    {
    switch (array.Type)
      {
    case DaxArray::IRREGULAR:
      return DaxArrayIrregular::GetScalar(work, array);
      }
    return -1;
    }

  __device__ static DaxVector3 GetVector3(const DaxWork& work, const DaxArray& array)
    {
    switch (array.Type)
      {
    case DaxArray::IRREGULAR:
      return DaxArrayIrregular::GetVector3(work, array);
      }
    return make_DaxVector3(0, 0, 0);
    }

  __device__ static DaxVector4 GetVector4(const DaxWork& work, const DaxArray& array)
    {
    switch (array.Type)
      {
    case DaxArray::IRREGULAR:
      return DaxArrayIrregular::GetVector4(work, array);
      }
    return make_DaxVector4(0, 0, 0, 0);
    }
};

#endif
