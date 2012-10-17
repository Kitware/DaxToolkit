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
#ifndef __dax_tbb_cont_internal_DeviceAdapterAlgorithmTBB_h
#define __dax_tbb_cont_internal_DeviceAdapterAlgorithmTBB_h

#include <dax/tbb/cont/internal/DeviceAdapterTagTBB.h>
#include <dax/tbb/cont/internal/ArrayManagerExecutionTBB.h>

#include <dax/exec/internal/ErrorMessageBuffer.h>

namespace dax {
namespace cont {
namespace internal {

namespace detail {

template<class FunctorType>
class ScheduleKernelTBB
{
public:
  DAX_CONT_EXPORT ScheduleKernelTBB(const FunctorType &functor)
    : Functor(functor)
  {  }

  DAX_CONT_EXPORT void SetErrorMessageBuffer(
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->ErrorMessage = errorMessage;
    this->Functor.SetErrorMessageBuffer(errorMessage);
  }

  DAX_EXEC_EXPORT void operator()(dax::Id index) const {
    // The TBB device adapter causes array classes to be shared between
    // control and execution environment. This means that it is possible for an
    // exception to be thrown even though this is typically not allowed.
    // Throwing an exception from here is bad because there are several
    // simultaneous threads running. Get around the problem by catching the
    // error and setting the message buffer as expected.
    try
      {
      this->Functor(index);
      }
    catch (dax::cont::Error error)
      {
      this->ErrorMessage.RaiseError(error.GetMessage().c_str());
      }
    catch (...)
      {
      this->ErrorMessage.RaiseError(
          "Unexpected error in execution environment.");
      }
  }

private:
  FunctorType Functor;
  dax::exec::internal::ErrorMessageBuffer ErrorMessage;
};

} // namespace detail

#if 0
template<class FunctorType>
DAX_CONT_EXPORT void LegacySchedule(
    FunctorType functor,
    dax::Id numInstances,
    dax::tbb::cont::DeviceAdapterTagTBB)
{
  dax::cont::internal::LegacySchedule(detail::ScheduleKernelTBB<FunctorType>(functor),
           numInstances,
           dax::thrust::cont::internal::DeviceAdapterTagThrust());
}
#endif

}
}
} // namespace dax::cont::internal

#endif //__dax_tbb_cont_internal_DeviceAdapterAlgorithmTBB_h
