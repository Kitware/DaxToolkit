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
#ifndef __dax_thrust_cont_internal_ArrayManagerExecutionThrustShare_h
#define __dax_thrust_cont_internal_ArrayManagerExecutionThrustShare_h

#include <dax/cont/internal/ArrayManagerExecutionShareWithControl.h>

namespace dax {
namespace thrust {
namespace cont {
namespace internal {

/// \c ArrayManagerExecutionThrustShare provides an implementation for a \c
/// ArrayManagerExecution class for a thrust device adapter that is designed
/// for a backend that has shared memory spaces for host and device.  This means
/// that, for example, a ::thrust::vectort_host and ::thrust::vector_device
/// allocate memory in the same space and can be used interchangeably.  This
/// is the case for backends such as OpenMP and TBB.  It is an error to use
/// this manager for backends like CUDA that require separate memory spaces.
///
template<typename T, class ArrayContainerControlTag>
class ArrayManagerExecutionThrustShare
    : public dax::cont::internal::ArrayManagerExecutionShareWithControl<T, ArrayContainerControlTag>
{
public:
  typedef dax::cont::internal::
      ArrayManagerExecutionShareWithControl<T, ArrayContainerControlTag>
        Superclass;
  typedef typename Superclass::ValueType ValueType;
  typedef typename Superclass::PortalType PortalType;
  typedef typename Superclass::PortalConstType PortalConstType;

  ArrayManagerExecutionThrustShare() {  }

  // These features are expected by thrust device adapters to run thrust
  // algorithms (see DeviceAdapterAlgorithmThrust.h).

  typedef typename PortalType::IteratorType ThrustIteratorType;
  typedef typename PortalConstType::IteratorType ThrustIteratorConstType;

  DAX_CONT_EXPORT static ThrustIteratorType
  ThrustIteratorBegin(PortalType portal) {
    return portal.GetIteratorBegin();
  }
  DAX_CONT_EXPORT static ThrustIteratorType
  ThrustIteratorEnd(PortalType portal) {
    return portal.GetIteratorEnd();
  }

  DAX_CONT_EXPORT static ThrustIteratorConstType
  ThrustIteratorBegin(PortalConstType portal) {
    return portal.GetIteratorBegin();
  }
  DAX_CONT_EXPORT static ThrustIteratorConstType
  ThrustIteratorEnd(PortalConstType portal) {
    return portal.GetIteratorEnd();
  }

private:
  // Not implemented
  ArrayManagerExecutionThrustShare(
      ArrayManagerExecutionThrustShare<T, ArrayContainerControlTag> &);
  void operator=(
      ArrayManagerExecutionThrustShare<T, ArrayContainerControlTag> &);
};

}
}
}
} // namespace dax::thrust::cont::internal

#endif //__dax_thrust_cont_internal_ArrayManagerExecutionThrustShare_h
