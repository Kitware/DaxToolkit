/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cuda_cont_DataArray_h
#define __dax_cuda_cont_DataArray_h


#include "Core/Common/DataArray.h"

namespace dax { namespace cuda { namespace cont {

  /// Creates a dax::core::DataArray instance on the device and copies the data
  /// from the host memory (referred by raw_data) to the device.
  dax::core::DataArray CreateAndCopyToDevice(
    dax::core::DataArray::eType type, dax::core::DataArray::eDataType dataType,
    unsigned int data_size_in_bytes, const void* raw_data);

  /// Creates a dax::core::DataArray instance on the device and allocates space
  /// for the memory requested on the device.
  dax::core::DataArray CreateOnDevice(
    dax::core::DataArray::eType type,
    dax::core::DataArray::eDataType dataType, unsigned int data_size_in_bytes);

  /// Copies data from device to host. Note that host memory should be
  /// pre-allocated.
  bool CopyToHost(const dax::core::DataArray& array,
    void* raw_data, unsigned int data_size_in_bytes);

}}}

#endif
