/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "DaxDataArray.h"
#include "assert.h"

//-----------------------------------------------------------------------------
DaxDataArray DaxDataArray::CreateAndCopy(
  eType type, eDataType dataType,
  unsigned int data_size_in_bytes, void* raw_data)
{
  DaxDataArray cur_array = Create(type, dataType, data_size_in_bytes);
  assert(cur_array.SizeInBytes == data_size_in_bytes);
  if (data_size_in_bytes > 0)
    {
    assert(cur_array.RawData != NULL);
    cudaMemcpy(cur_array.RawData, raw_data, data_size_in_bytes,
      cudaMemcpyHostToDevice);
    }
  return cur_array;
}
//-----------------------------------------------------------------------------
bool DaxDataArray::CopyTo(void* raw_data, unsigned int data_size_in_bytes) const
{
  assert(this->SizeInBytes >= data_size_in_bytes);
  if (data_size_in_bytes > 0)
    {
    assert(this->RawData != NULL && raw_data != NULL);
    cudaMemcpy(raw_data, this->RawData, data_size_in_bytes,
      cudaMemcpyDeviceToHost);
    }
  return true;
}

//-----------------------------------------------------------------------------
DaxDataArray DaxDataArray::Create(
  eType type, eDataType dataType, unsigned int data_size_in_bytes)
{
  DaxDataArray cur_array;
  cur_array.Type = type;
  cur_array.DataType = dataType;
  cur_array.SizeInBytes = data_size_in_bytes;
  cur_array.RawData = NULL;
  if (data_size_in_bytes > 0)
    {
    cudaMalloc(&cur_array.RawData, data_size_in_bytes);
    assert(cur_array.RawData != NULL);
    }
  return cur_array;
}

