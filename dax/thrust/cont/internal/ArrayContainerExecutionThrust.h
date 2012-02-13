/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_thrust_cont_internal_ArrayContainerExecutionThrust_h
#define __dax_thrust_cont_internal_ArrayContainerExecutionThrust_h

#include <dax/Types.h>

#include <dax/internal/DataArray.h>

#include <dax/cont/Assert.h>
#include <dax/cont/ErrorControlOutOfMemory.h>
#include <dax/cont/internal/IteratorContainer.h>

#include <thrust/device_vector.h>

namespace dax {
namespace thrust {
namespace cont {
namespace internal {

/// Manages a Thrust device array. Can allocate the array of the given type on
/// the device, copy data do and from it, and release the memory. The memory is
/// also released when this object goes out of scope.
///
template<typename T>
class ArrayContainerExecutionThrust
{
public:
  typedef T ValueType;

  /// On inital creation, no memory is allocated on the device.
  ///
  ArrayContainerExecutionThrust() { }

  /// Allocates an array on the device large enough to hold the given number of
  /// entries.
  ///
  void Allocate(dax::Id numEntries) {
    try
      {
      this->DeviceArray.resize(numEntries);
      }
    catch (...)
      {
      throw dax::cont::ErrorControlOutOfMemory(
          "Failed to allocate execution array with thrust.");
      }
  }

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
  typename ::thrust::device_vector<ValueType>::iterator GetBeginThrustIterator()
  {
    return this->DeviceArray.begin();
  }
  typename ::thrust::device_vector<ValueType>::iterator GetEndThrustIterator()
  {
    return this->DeviceArray.end();
  }

  /// Returns a DataArray structure for the array on the device.  This array
  /// can be passed to a device backend kernel.
  ///
  dax::internal::DataArray<ValueType> GetExecutionArray();

  /// Allows you to use this class like an array. However, accessing
  /// independent locations might not be efficient.
  ///
  ::thrust::device_reference<ValueType> operator[](dax::Id index);
  const ::thrust::device_reference<ValueType> operator[](dax::Id index) const;

private:
  ArrayContainerExecutionThrust(const ArrayContainerExecutionThrust &); // Not implemented
  void operator=(const ArrayContainerExecutionThrust &);     // Not implemented

  ::thrust::device_vector<ValueType> DeviceArray;
};

//-----------------------------------------------------------------------------
template<class T>
template<class IteratorType>
inline void ArrayContainerExecutionThrust<T>::CopyFromControlToExecution(
    const dax::cont::internal::IteratorContainer<IteratorType> &iterators)
{
  DAX_ASSERT_CONT(iterators.IsValid());
  DAX_ASSERT_CONT(iterators.GetNumberOfEntries()
                  == static_cast<dax::Id>(this->DeviceArray.size()));
  ::thrust::copy(iterators.GetBeginIterator(),
                 iterators.GetEndIterator(),
                 this->DeviceArray.begin());
}

//-----------------------------------------------------------------------------
template<class T>
template<class IteratorType>
inline void ArrayContainerExecutionThrust<T>::CopyFromExecutionToControl(
    const dax::cont::internal::IteratorContainer<IteratorType> &iterators)
{
  DAX_ASSERT_CONT(iterators.IsValid());
  DAX_ASSERT_CONT(iterators.GetNumberOfEntries()
                  == static_cast<dax::Id>(this->DeviceArray.size()));
  ::thrust::copy(this->DeviceArray.begin(),
                 this->DeviceArray.end(),
                 iterators.GetBeginIterator());
}

//-----------------------------------------------------------------------------
template<class T>
inline dax::internal::DataArray<T>
ArrayContainerExecutionThrust<T>::GetExecutionArray()
{
  ValueType *rawPointer = ::thrust::raw_pointer_cast(&this->DeviceArray[0]);
  dax::Id numEntries = this->DeviceArray.size();
  return dax::internal::DataArray<ValueType>(rawPointer, numEntries);
}

//-----------------------------------------------------------------------------
template<class T>
inline ::thrust::device_reference<T>
ArrayContainerExecutionThrust<T>::operator[](dax::Id index)
{
  DAX_ASSERT_CONT(index >= 0);
  DAX_ASSERT_CONT(index < static_cast<dax::Id>(this->DeviceArray.size()));
  return this->DeviceArray[index];
}

template<class T>
inline const ::thrust::device_reference<T>
ArrayContainerExecutionThrust<T>::operator[](dax::Id index) const
{
  DAX_ASSERT_CONT(index >= 0);
  DAX_ASSERT_CONT(index < static_cast<dax::Id>(this->DeviceArray.size()));
  return this->DeviceArray[index];
}

}
}
}
}

#endif // __dax_thrust_cont_internal_ArrayContainerExecutionThrust_h
