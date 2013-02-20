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
#ifndef __dax_openmp_cont_internal_DeviceAdapterAlgorithmOpenMP_h
#define __dax_openmp_cont_internal_DeviceAdapterAlgorithmOpenMP_h

#include <dax/openmp/cont/internal/SetThrustForOpenMP.h>

#include <dax/openmp/cont/internal/DeviceAdapterTagOpenMP.h>
#include <dax/openmp/cont/internal/ArrayManagerExecutionOpenMP.h>

#include <dax/cont/internal/DeviceAdapterAlgorithm.h>

// Here are the actual implementation of the algorithms.
#include <dax/thrust/cont/internal/DeviceAdapterAlgorithmThrust.h>

namespace dax {
namespace cont {
namespace internal {

template<>
struct DeviceAdapterAlgorithm<dax::openmp::cont::DeviceAdapterTagOpenMP>
    : public dax::thrust::cont::internal::DeviceAdapterAlgorithmThrust<
          dax::openmp::cont::DeviceAdapterTagOpenMP>
{
private:
  typedef dax::thrust::cont::internal::DeviceAdapterAlgorithmThrust<
      dax::openmp::cont::DeviceAdapterTagOpenMP> Superclass;

  template<class FunctorType>
  class ScheduleKernel
  {
  public:
    DAX_CONT_EXPORT ScheduleKernel(const FunctorType &functor)
      : Functor(functor)
    {  }

    DAX_CONT_EXPORT void SetErrorMessageBuffer(
        dax::exec::internal::ErrorMessageBuffer &errorMessage)
    {
      this->ErrorMessage = errorMessage;
      this->Functor.SetErrorMessageBuffer(errorMessage);
    }

    DAX_EXEC_EXPORT void operator()(dax::Id index) const {
      // The OpenMP device adapter causes array classes to be shared between
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

public:
  // Override the thrust version of Schedule to handle exceptions that can occur
  // because we are running on a CPU.
  template<class FunctorType>
  DAX_CONT_EXPORT
  static void Schedule(FunctorType functor, dax::Id numInstances)
  {
     Superclass::Schedule(
           DeviceAdapterAlgorithm::ScheduleKernel<FunctorType>(functor),
           numInstances);
  }

  template<class FunctorType>
  DAX_CONT_EXPORT
  static void Schedule(FunctorType functor, dax::Id3 rangeMax)
  {
    //default behavior for the general algorithm is to defer to the default
    //schedule implementation.
    Superclass::Schedule(
           DeviceAdapterAlgorithm::ScheduleKernel<FunctorType>(functor),
           rangeMax);
  }

};

}
}
} // namespace dax::cont::internal

#endif //__dax_openmp_cont_internal_DeviceAdapterAlgorithmOpenMP_h
