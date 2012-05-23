//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_thrust_cont_internal_ArrayManagerExecutionThrustDevice_h
#define __dax_thrust_cont_internal_ArrayManagerExecutionThrustDevice_h

#include <dax/cont/ArrayContainerControl.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

namespace dax {
namespace thrust {
namespace cont {
namespace internal {

/// \c ArrayManagerExecutionThrustDevice provides an implementation for a \c
/// ArrayManagerExecution class for a thrust device adapter that is designed
/// for a backend that has separate memory spaces for host and device. This
/// implementation contains a ::thrust::device_vector to allocate and manage
/// the array.
///
/// This array manager can be used for any thrust-based device adapter, but it
/// really should only be used when the host and device use different memory
/// spaces (such as a CUDA backend). If they share the same memory space (such
/// as an OpenMP or TBB backend), then the device adapter should just use
/// ArrayManagerExecutionThrustShare.
///
template<typename T, class ArrayContainerControlTag>
class ArrayManagerExecutionThrustDevice
{
public:
  typedef T ValueType;
  typedef dax::cont::internal
      ::ArrayContainerControl<ValueType, ArrayContainerControlTag>
      ContainerType;
  typedef ValueType *IteratorType;
  typedef const ValueType *IteratorConstType;

  DAX_CONT_EXPORT ArrayManagerExecutionThrustDevice() {  }

  /// Allocates the appropriate size of the array and copies the given data
  /// into the array.
  ///
  template <class IteratorTypeControl>
  DAX_CONT_EXPORT void LoadDataForInput(IteratorTypeControl beginIterator,
                                        IteratorTypeControl endIterator)
  {
    this->Array.assign(beginIterator, endIterator);
  }

  /// Allocates the array to the given size.
  ///
  DAX_CONT_EXPORT void AllocateArrayForOutput(ContainerType &,
                                              dax::Id numberOfValues)
  {
    // This call is a bit wasteful in that it sets all the values of the
    // array to the default value.
    this->Array.resize(numberOfValues);
  }

  /// Copies the data currently in the device array into the given iterators.
  /// Although the iterator is supposed to be from the control environment,
  /// thrust can generally handle iterators for a device as well.
  ///
  template <class IteratorTypeControl>
  DAX_CONT_EXPORT void CopyInto(IteratorTypeControl dest) const
  {
    ::thrust::copy(this->Array.cbegin(), this->Array.cend(), dest);
  }

  /// Allocates enough space in \c controlArray and copies the data in the
  /// device vector into it.
  ///
  DAX_CONT_EXPORT void RetrieveOutputData(ContainerType &controlArray) const
  {
    controlArray.Allocate(this->Array.size());
    this->CopyInto(controlArray.GetIteratorBegin());
  }

  /// Resizes the device vector.
  ///
  DAX_CONT_EXPORT void Shrink(dax::Id numberOfValues)
  {
    // The operation will succeed even if this assertion fails, but this
    // is still supposed to be a precondition to Shrink.
    DAX_ASSERT_CONT(numberOfValues <= static_cast<dax::Id>(this->Array.size()));

    this->Array.resize(numberOfValues);
  }

  DAX_CONT_EXPORT IteratorType GetIteratorBegin()
  {
    return this->ThrustIteratorToExecutionIterator(this->Array.begin());
  }
  DAX_CONT_EXPORT IteratorType GetIteratorEnd()
  {
    return this->ThrustIteratorToExecutionIterator(this->Array.end());
  }
  DAX_CONT_EXPORT IteratorConstType GetIteratorConstBegin() const
  {
    return this->ThrustIteratorToExecutionIterator(this->Array.cbegin());
  }
  DAX_CONT_EXPORT IteratorConstType GetIteratorConstEnd() const
  {
    return this->ThrustIteratorToExecutionIterator(this->Array.cend());
  }

  /// Frees all memory.
  ///
  DAX_CONT_EXPORT void ReleaseResources() {
    this->Array.clear();
    this->Array.shrink_to_fit();
  }

  // These features are expected by thrust device adapters to run thrust
  // algorithms (see DeviceAdapterThrust.h).

  typedef typename ::thrust::device_vector<ValueType>::iterator
      ThrustIteratorType;
  typedef typename ::thrust::device_vector<ValueType>::const_iterator
      ThrustIteratorConstType;

  ThrustIteratorType GetThrustIteratorBegin() { return this->Array.begin(); }
  ThrustIteratorType GetThrustIteratorEnd() { return this->Array.end(); }

  ThrustIteratorConstType GetThrustIteratorConstBegin() const {
    return this->Array.cbegin();
  }
  ThrustIteratorConstType GetThrustIteratorConstEnd() const {
    return this->Array.cend();
  }

private:
  // Not implemented
  ArrayManagerExecutionThrustDevice(
      ArrayManagerExecutionThrustDevice<T, ArrayContainerControlTag> &);
  void operator=(
      ArrayManagerExecutionThrustDevice<T, ArrayContainerControlTag> &);

  ::thrust::device_vector<ValueType> Array;

  // For technical reasons, Dax does not use thrust iterators outside these
  // limited contexts. These methods convert a thrust iterator to an iterator
  // used in a Dax execution environment.
  DAX_CONT_EXPORT static IteratorType ThrustIteratorToExecutionIterator(
      ThrustIteratorType thrustIterator)
  {
    return ::thrust::raw_pointer_cast(&(*thrustIterator));
  }
  DAX_CONT_EXPORT static IteratorConstType ThrustIteratorToExecutionIterator(
      ThrustIteratorConstType thrustIterator)
  {
    return ::thrust::raw_pointer_cast(&(*thrustIterator));
  }
};

}
}
}
} // namespace dax::thrust::cont::internal

#endif // __dax_thrust_cont_internal_ArrayManagerExecutionThrustDevice_h
