/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cuda_cont_internal_ArrayContainerExecution_h
#define __dax_cuda_cont_internal_ArrayContainerExecution_h

#include <dax/Types.h>

#include <dax/internal/DataArray.h>

#include <dax/cont/internal/IteratorContainer.h>

#include <thrust/device_vector.h>

#include <assert.h>

namespace dax {
namespace cuda {
namespace cont {
namespace internal {

/// Manages a CUDA device array. Can allocate the array of the given type on
/// the device, copy data do and from it, and release the memory. The memory is
/// also released when this object goes out of scope.
///
template<typename T>
class ArrayContainerExecution
{
public:
  typedef T ValueType;

  /// On inital creation, no memory is allocated on the device.
  ///
  ArrayContainerExecution() { }

  /// Allocates an array on the device large enough to hold the given number of
  /// entries.
  ///
  void Allocate(dax::Id numEntries) { this->DeviceArray.resize(numEntries); }

  /// Copies the data pointed to by the passed in \c iterators (assumed to be
  /// in the control environment), into the array in the execution environment
  /// managed by this class.
  ///
  template<class IteratorType>
  void CopyFromControlToExecution(
      const dax::cont::internal::IteratorContainer<IteratorType> &iterators);

  /// Copies the data from the array in the execution environment managed by
  /// this class into the memory passed in the \c iterators (assumed to be in
  /// the control environment).
  ///
  template<class IteratorType>
  void CopyFromExecutionToControl(
      const dax::cont::internal::IteratorContainer<IteratorType> &iterators);

  /// Frees any resources (i.e. memory) on the device.
  ///
  void ReleaseResources() { this->Allocate(0); }

  /// Gets the thrust iterators for the contained array. May be too low level
  /// to expose to everyone
  ///
  typename thrust::device_vector<ValueType>::iterator GetBeginThrustIterator() {
    return this->DeviceArray.begin();
  }
  typename thrust::device_vector<ValueType>::iterator GetEndThrustIterator() {
    return this->DeviceArray.end();
  }

  /// Returns a DataArray structure for the array on the device.  This array
  /// can be passed to a CUDA kernel.
  ///
  dax::internal::DataArray<ValueType> GetExecutionArray();

private:
  ArrayContainerExecution(const ArrayContainerExecution &); // Not implemented
  void operator=(const ArrayContainerExecution &);          // Not implemented

  thrust::device_vector<ValueType> DeviceArray;
};

//-----------------------------------------------------------------------------
template<class T>
template<class IteratorType>
inline void ArrayContainerExecution<T>::CopyFromControlToExecution(
    const dax::cont::internal::IteratorContainer<IteratorType> &iterators)
{
  assert(iterators.IsValid());
  assert(iterators.GetNumberOfEntries()
         <= static_cast<dax::Id>(this->DeviceArray.size()));
  thrust::copy(iterators.GetBeginIterator(),
               iterators.GetEndIterator(),
               this->DeviceArray.begin());
}

//-----------------------------------------------------------------------------
template<class T>
template<class IteratorType>
inline void ArrayContainerExecution<T>::CopyFromExecutionToControl(
    const dax::cont::internal::IteratorContainer<IteratorType> &iterators)
{
  assert(iterators.IsValid());
  assert(iterators.GetNumberOfEntries()
         <= static_cast<dax::Id>(this->DeviceArray.size()));
  thrust::copy(this->DeviceArray.begin(),
               this->DeviceArray.end(),
               iterators.GetBeginIterator());
}

//-----------------------------------------------------------------------------
template<class T>
inline dax::internal::DataArray<T>
ArrayContainerExecution<T>::GetExecutionArray()
{
  ValueType *rawPointer = thrust::raw_pointer_cast(&this->DeviceArray[0]);
  dax::Id numEntries = this->DeviceArray.size();
  return dax::internal::DataArray<ValueType>(rawPointer, numEntries);
}

}
}
}
}

#endif // __dax_cuda_cont_internal_ArrayContainerExecution_h
