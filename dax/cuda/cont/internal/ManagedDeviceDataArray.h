/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cuda_cont_internal_ManagedDeviceDataArray_h
#define __dax_cuda_cont_internal_ManagedDeviceDataArray_h


#include <dax/cont/internal/Object.h>

#include <dax/internal/DataArray.h>

#include <cuda.h>

#include <assert.h>

namespace dax {
namespace cuda {
namespace cont {
namespace internal {

daxDeclareClassTemplate1(ManagedDeviceDataArray);

template<typename T>
class ManagedDeviceDataArray : public dax::cont::internal::Object
{
public:
  typedef T ValueType;
  static const dax::Id ValueSize = sizeof(ValueType);

  inline ManagedDeviceDataArray() : Array() { }

  void Allocate(dax::Id numEntries)
  {
    if (numEntries == this->Array.GetNumberOfEntries()) { return; }

    this->Free();

    dax::Id sizeInBytes = numEntries * ManagedDeviceDataArray::ValueSize;

    T *buffer;
    cudaMalloc(&buffer, sizeInBytes);
    assert(buffer != NULL);

    this->Array.SetPointer(buffer, numEntries);
  }

  void Free()
  {
    if (this->Array.GetNumberOfEntries() > 0)
      {
      cudaError_t error;
      error = cudaFree(this->Array.GetPointer());
      assert(error == cudaSuccess);
      this->Array.SetPointer(NULL, 0);
      }
  }

  void CopyToDevice(const dax::internal::DataArray<ValueType> &srcArray)
  {
    dax::Id numEntries = srcArray.GetNumberOfEntries();
    this->Allocate(numEntries);

    dax::Id sizeInBytes = numEntries * ManagedDeviceDataArray::ValueSize;

    cudaError_t error;
    error = cudaMemcpy(this->Array.GetPointer(),
                       srcArray.GetPointer(),
                       sizeInBytes,
                       cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);
  }

  void CopyToHost(dax::internal::DataArray<ValueType> &destArray) const
  {
    dax::Id numEntries = this->Array.GetNumberOfEntries();
    // TODO: Better error checking
    assert(numEntries == destArray.GetNumberOfEntries());

    dax::Id sizeInBytes = numEntries * ManagedDeviceDataArray::ValueSize;

    cudaError_t error;
    error = cudaMemcpy(destArray.GetPointer(),
                       this->Array.GetPointer(),
                       sizeInBytes,
                       cudaMemcpyDeviceToHost);
    assert(error == cudaSuccess);
  }

  dax::internal::DataArray<ValueType> GetArray() { return this->Array; }

private:
  dax::internal::DataArray<ValueType> Array;

  daxDisableCopyMacro(ManagedDeviceDataArray);
};

}}}}

#endif //__dax_cuda_cont_internal_ManagedDeviceDataArray_h
