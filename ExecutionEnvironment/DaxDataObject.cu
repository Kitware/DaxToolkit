/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/// This file defines the DataObject that can be used on the host as well as
/// OnDevice.

#include "DaxCommon.h"

class DaxArray
{
public:
  enum eType
    {
    UNKNOWN,
    IRREGULAR,
    STRUCTURED_POINTS,
    STRUCTURED_CELLS,
    };

  eType Type;
  float* RawData;
  DaxId Size;
  bool OnDevice;

  __device__ __host__ DaxArray()
    { 
    this->Type = UNKNOWN;
    this->RawData = NULL;
    this->OnDevice = false;
    this->Size = 0;
    }

  __host__ void CopyFrom(const DaxArray& source)
    {
    this->Allocate(source);
    if (!this->OnDevice  && !source.OnDevice)
      {
      memcpy(this->RawData, source.RawData, source.Size*sizeof(float));
      }
    else if (this->OnDevice)
      {
      cudaMemcpy(this->RawData, source.RawData,
        source.Size*sizeof(float), cudaMemcpyHostToDevice);
      }
    else if (source.OnDevice)
      {
      cudaMemcpy(this->RawData, source.RawData,
        this->Size * sizeof(float), cudaMemcpyDeviceToHost);
      }
    }

  __host__ void Allocate(const DaxArray& source)
    {
    this->FreeMemory();
    if (this->OnDevice)
      {
      cudaMalloc(&this->RawData, source.Size * sizeof(float));
      }
    else
      {
      this->RawData = new float[source.Size];
      }
    this->Size = source.Size;
    }

  __host__ void FreeMemory()
    {
    if (this->OnDevice)
      {
      if (this->RawData && this->Size)
        {
        cudaFree(this->RawData);
        }
      }
    else
      {
      delete [] this->RawData;
      }
    this->RawData = NULL;
    this->Size = 0;
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
    delete [] this->RawData;
    this->RawData = new float[this->NumberOfComponents * this->NumberOfTuples];
    this->Size = this->NumberOfTuples * this->NumberOfComponents;
    }

  __host__ void SetValue(DaxId tupleId, DaxId componentId, float value)
    {
    this->RawData[tupleId * this->NumberOfComponents + componentId] = value;
    }
  __host__ float GetValue(DaxId tupleId, DaxId componentId)
    {
    return this->RawData[tupleId * this->NumberOfComponents + componentId];
    }
};

class DaxDataObject
{
public:
  DaxArray PointData;
  DaxArray CellData;
  DaxArray PointCoordinates;
  DaxArray CellArray;

  __host__ void FreeMemory()
    {
    this->PointData.FreeMemory();
    this->CellData.FreeMemory();
    this->PointCoordinates.FreeMemory();
    this->CellArray.FreeMemory();
    }

  __host__ void CopyFrom(const DaxDataObject& source)
    {
    this->PointData.CopyFrom(source.PointData);
    this->CellData.CopyFrom(source.CellData);
    this->PointCoordinates.CopyFrom(source.PointCoordinates);
    this->CellArray.CopyFrom(source.CellArray);
    }

  __host__ void Allocate(const DaxDataObject& source)
    {
    this->PointData.Allocate(source.PointData);
    this->CellData.Allocate(source.CellData);
    this->PointCoordinates.Allocate(source.PointCoordinates);
    this->CellArray.Allocate(source.CellArray);
    }
};

class DaxDataObjectDevice : public DaxDataObject
{
public:
  __host__ DaxDataObjectDevice()
    {
    this->PointData.OnDevice = true;
    this->CellData.OnDevice = true;
    this->PointCoordinates.OnDevice = true;
    this->CellArray.OnDevice = true;
    }
};
