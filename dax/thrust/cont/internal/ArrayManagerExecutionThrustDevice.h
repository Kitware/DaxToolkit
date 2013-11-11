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

#include <dax/thrust/cont/internal/CheckThrustBackend.h>

#include <dax/cont/ArrayContainerControl.h>
#include <dax/cont/ErrorControlOutOfMemory.h>

#include <dax/exec/internal/ArrayPortalFromIterators.h>

// Disable GCC warnings we check Dax for but Thrust does not.
#if defined(__GNUC__) && !defined(DAX_CUDA)
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#endif // gcc version >= 4.6
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 2)
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif // gcc version >= 4.2
#endif // gcc && !CUDA

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#if defined(__GNUC__) && !defined(DAX_CUDA)
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 6)
#pragma GCC diagnostic pop
#endif // gcc version >= 4.6
#endif // gcc && !CUDA

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
  typedef dax::exec::internal::ArrayPortalFromIterators<ValueType *>
      PortalType;
  typedef dax::exec::internal::ArrayPortalFromIterators<const ValueType *>
      PortalConstType;

  DAX_CONT_EXPORT ArrayManagerExecutionThrustDevice() {  }

  /// Returns the size of the array.
  ///
  DAX_CONT_EXPORT dax::Id GetNumberOfValues() const {
    return this->Array.size();
  }

  /// Allocates the appropriate size of the array and copies the given data
  /// into the array.
  ///
  template<class PortalControl>
  DAX_CONT_EXPORT void LoadDataForInput(PortalControl arrayPortal)
  {
    try
      {
      this->Array.assign(arrayPortal.GetIteratorBegin(),
                         arrayPortal.GetIteratorEnd());
      }
    catch (std::bad_alloc error)
      {
      throw dax::cont::ErrorControlOutOfMemory(error.what());
      }
  }

  /// Allocates the appropriate size of the array and copies the given data
  /// into the array.
  ///
  DAX_CONT_EXPORT void LoadDataForInPlace(ContainerType &controlArray)
  {
    this->LoadDataForInput(controlArray.GetPortalConst());
  }

  /// Allocates the array to the given size.
  ///
  DAX_CONT_EXPORT void AllocateArrayForOutput(
      ContainerType &daxNotUsed(container),
      dax::Id numberOfValues)
  {
    try
      {
      // This call is a bit wasteful in that it sets all the values of the
      // array to the default value.
      this->Array.resize(numberOfValues);
      }
    catch (std::bad_alloc error)
      {
      throw dax::cont::ErrorControlOutOfMemory(error.what());
      }
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
    this->CopyInto(controlArray.GetPortal().GetIteratorBegin());
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

  DAX_CONT_EXPORT PortalType GetPortal()
  {
    return PortalType(::thrust::raw_pointer_cast(&(*this->Array.begin())),
                      ::thrust::raw_pointer_cast(&(*this->Array.end())));
  }

  DAX_CONT_EXPORT PortalConstType GetPortalConst() const
  {
    return PortalConstType(::thrust::raw_pointer_cast(&(*this->Array.cbegin())),
                           ::thrust::raw_pointer_cast(&(*this->Array.cend())));
  }

  /// Frees all memory.
  ///
  DAX_CONT_EXPORT void ReleaseResources() {
    this->Array.clear();
    this->Array.shrink_to_fit();
  }

  // These features are expected by thrust device adapters to run thrust
  // algorithms (see DeviceAdapterAlgorithmThrust.h).

  typedef ::thrust::device_ptr<ValueType> ThrustIteratorType;
  typedef ::thrust::device_ptr<const ValueType> ThrustIteratorConstType;

  DAX_CONT_EXPORT static ThrustIteratorType
  ThrustIteratorBegin(PortalType portal) {
    return ::thrust::device_ptr<ValueType>(portal.GetIteratorBegin());
  }
  DAX_CONT_EXPORT static ThrustIteratorType
  ThrustIteratorEnd(PortalType portal) {
    return ::thrust::device_ptr<ValueType>(portal.GetIteratorEnd());
  }
  DAX_CONT_EXPORT static ThrustIteratorConstType
  ThrustIteratorBegin(PortalConstType portal) {
    return ::thrust::device_ptr<const ValueType>(portal.GetIteratorBegin());
  }
  DAX_CONT_EXPORT static ThrustIteratorConstType
  ThrustIteratorEnd(PortalConstType portal) {
    return ::thrust::device_ptr<const ValueType>(portal.GetIteratorEnd());
  }

private:
  // Not implemented
  ArrayManagerExecutionThrustDevice(
      ArrayManagerExecutionThrustDevice<T, ArrayContainerControlTag> &);
  void operator=(
      ArrayManagerExecutionThrustDevice<T, ArrayContainerControlTag> &);

  ::thrust::device_vector<ValueType> Array;
};

}
}
}
} // namespace dax::thrust::cont::internal

#endif // __dax_thrust_cont_internal_ArrayManagerExecutionThrustDevice_h
