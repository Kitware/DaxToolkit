/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cuda_cont_internal_DataArray_h
#define __dax_cuda_cont_internal_DataArray_h


#include <dax/internal/DataArray.h>

namespace dax { namespace cuda { namespace cont { namespace internal {

  /// Creates a dax::core::DataArray instance on the device and copies the data
  /// from the host memory (referred by raw_data) to the device.
  dax::internal::DataArray CreateAndCopyToDevice(
    dax::internal::DataArray::eType type, dax::internal::DataArray::eDataType dataType,
    unsigned int data_size_in_bytes, const void* raw_data);

  /// Creates a dax::core::DataArray instance on the device and allocates space
  /// for the memory requested on the device.
  dax::internal::DataArray CreateOnDevice(
    dax::internal::DataArray::eType type,
    dax::internal::DataArray::eDataType dataType, unsigned int data_size_in_bytes);

  /// Copies data from device to host. Note that host memory should be
  /// pre-allocated.
  bool CopyToHost(const dax::internal::DataArray& array,
    void* raw_data, unsigned int data_size_in_bytes);

}}}}

#endif
