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

#include <omp.h>

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

  DAX_CONT_EXPORT static void Synchronize()
  {
    // Nothing to do. This OpenMP schedules all of its operations using a
    // split/join paradigm. This means that the if the control threaad is
    // calling this method, then nothing should be running in the execution
    // environment.
  }

};

}
}
} // namespace dax::cont::internal

namespace dax {
namespace cont {

// Add prototype for Timer template, which might not be defined yet.
template<class DeviceAdapter> class Timer;

/// OpenMP contains its own high resolution timer.
///
template<>
class Timer<dax::openmp::cont::DeviceAdapterTagOpenMP>
{
public:
  DAX_CONT_EXPORT Timer()
  {
    this->Reset();
  }
  DAX_CONT_EXPORT void Reset()
  {
    dax::cont::internal::DeviceAdapterAlgorithm<
        dax::openmp::cont::DeviceAdapterTagOpenMP>::Synchronize();
    this->StartTime = omp_get_wtime();
  }
  DAX_CONT_EXPORT dax::Scalar GetElapsedTime()
  {
    dax::cont::internal::DeviceAdapterAlgorithm<
        dax::openmp::cont::DeviceAdapterTagOpenMP>::Synchronize();
    double currentTime = omp_get_wtime();
    return static_cast<dax::Scalar>(currentTime - this->StartTime);
  }

private:
  double StartTime;
};

}
} // namespace dax::cont

#endif //__dax_openmp_cont_internal_DeviceAdapterAlgorithmOpenMP_h
