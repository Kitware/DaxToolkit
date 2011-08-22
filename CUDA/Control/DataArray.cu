/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "CUDA/Control/DataArray.h"


#include <assert.h>

//-----------------------------------------------------------------------------
dax::core::DataArray dax::cuda::cont::CreateAndCopyToDevice(
  dax::core::DataArray::eType type, dax::core::DataArray::eDataType dataType,
  unsigned int data_size_in_bytes, const void* raw_data)
{
  dax::core::DataArray cur_array =
    dax::cuda::cont::CreateOnDevice(type, dataType, data_size_in_bytes);
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
bool dax::cuda::cont::CopyToHost(const dax::core::DataArray& array,
  void* raw_data, unsigned int data_size_in_bytes)
{
  assert(array.SizeInBytes >= data_size_in_bytes);
  if (data_size_in_bytes > 0)
    {
    assert(array.RawData != NULL && raw_data != NULL);
    cudaMemcpy(raw_data, array.RawData, data_size_in_bytes,
      cudaMemcpyDeviceToHost);
    }
  return true;
}

//-----------------------------------------------------------------------------
dax::core::DataArray dax::cuda::cont::CreateOnDevice(
  dax::core::DataArray::eType type,
  dax::core::DataArray::eDataType dataType, unsigned int data_size_in_bytes)
{
  dax::core::DataArray cur_array;
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

