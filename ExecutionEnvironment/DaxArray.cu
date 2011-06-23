/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __DaxArray_h
#define __DaxArray_h

#include "DaxCommon.h"
#include "DaxWork.cu"

#include <assert.h>

/// DaxArray is a basic data-storage device in Dax data model. It stores the
/// heavy data. A dataset comprises for DaxArray instances assigned different
/// roles, for example an "array" for storing point coordinates, an "array" for
/// cell-connectivity etc. Different types of arrays exist. The subclasses are
/// used in control environment to define the datasets. In the execution
/// environment, user code i.e the worklet should never use DaxArray directly
/// (it should rely on DaxField or subclasses). The execution environment uses
/// various traits to access raw-data from the DaxArray.
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
    if (size_in_bytes > 0)
      {
      if (this->OnDevice)
        {
        cudaMalloc(&this->RawData, size_in_bytes);
        }
      else
        {
        this->RawData = malloc(size_in_bytes);
        }
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

#endif
