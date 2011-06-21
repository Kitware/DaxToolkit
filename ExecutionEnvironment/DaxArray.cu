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
    FLOAT,
    INT
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
};

class DaxArrayIrregular : public DaxArray
{
  SUPERCLASS(DaxArray);
  DaxId NumberOfTuples;
  DaxId NumberOfComponents;
public:
  __host__ DaxArrayIrregular() :
    NumberOfTuples(0),
    NumberOfComponents(1)
    {
    this->Type = IRREGULAR;
    // maybe we use templates?
    this->DataType = FLOAT;
    }

  __host__ void SetNumberOfTuples(DaxId val)
    {
    this->NumberOfTuples = val;
    }

  __host__ void SetNumberOfComponents(DaxId val)
    {
    this->NumberOfComponents = val;
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
  friend class DaxArraySetterTraits;

  __device__ static void Set(const DaxWork& work, DaxArray& array, DaxScalar scalar)
    {
    reinterpret_cast<float*>(array.RawData)[work.GetItem()] = scalar;
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
    this->DataType = INT;
    }

  __device__ static DaxId GetNumberOfConnectedElements(
    const DaxWork&, const DaxArray&)
    {
    return 8;
    }

  __device__ static DaxWorkMapField GetPoint(const DaxWork& work,
    const DaxArray& cellArray)
    {
    MetadataType* metadata = reinterpret_cast<MetadataType*>(
      cellArray.RawData);

    DaxId flat_id = work.GetItem();
    // given the flat_id, what is the ijk value?
    int3 dims;
    dims.x = metadata->ExtentMax.x - metadata->ExtentMin.x + 1;
    dims.y = metadata->ExtentMax.y - metadata->ExtentMin.y + 1;
    dims.z = metadata->ExtentMax.z - metadata->ExtentMin.z + 1;

    int3 ijk;
    ijk.x = flat_id % (dims.x -1);
    ijk.y = (flat_id / (dims.x - 1)) % (dims.y -1);
    ijk.z = (flat_id / ((dims.x-1) * (dims.y -1)));
    return DaxWorkMapField();
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

};

#endif
