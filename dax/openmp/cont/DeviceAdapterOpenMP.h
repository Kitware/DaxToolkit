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

#ifndef __dax_openmp_cont_DeviceAdapterOpenMP_h
#define __dax_openmp_cont_DeviceAdapterOpenMP_h

#ifdef DAX_DEFAULT_DEVICE_ADAPTER
#undef DAX_DEFAULT_DEVICE_ADAPTER
#endif

#define DAX_DEFAULT_DEVICE_ADAPTER ::dax::openmp::cont::DeviceAdapterOpenMP

#include <dax/openmp/cont/Copy.h>
#include <dax/openmp/cont/InclusiveScan.h>
#include <dax/openmp/cont/LowerBounds.h>
#include <dax/openmp/cont/ScheduleThrust.h>
#include <dax/openmp/cont/StreamCompact.h>
#include <dax/openmp/cont/Sort.h>
#include <dax/openmp/cont/Unique.h>
#include <dax/openmp/cont/internal/ArrayContainerExecutionThrust.h>

namespace dax {
namespace cont {
  //forward declare the ArrayHandle before we use it.
  template< typename OtherT, class OtherDeviceAdapter > class ArrayHandle;
}
}

namespace dax {
namespace openmp {
namespace cont {

/// An implementation of DeviceAdapter that will schedule execution on multiple
/// CPUs using OpenMP.
///
struct DeviceAdapterOpenMP
{
  template<typename T>
  class ArrayContainerExecution
      : public dax::openmp::cont::internal::ArrayContainerExecutionThrust<T>
  { };

  template<typename T>
  static void Copy(const dax::cont::ArrayHandle<T,DeviceAdapterOpenMP>& from,
                         dax::cont::ArrayHandle<T,DeviceAdapterOpenMP>& to)
    {
    DAX_ASSERT_CONT(from.hasExecutionArray());
    DAX_ASSERT_CONT(to.GetNumberOfEntries() >= from.GetNumberOfEntries());
    to.ReadyAsOutput();
    dax::openmp::cont::copy(from.GetExecutionArray(),to.GetExecutionArray());
    }

  template<typename T>
  static T InclusiveScan(const dax::cont::ArrayHandle<T,DeviceAdapterOpenMP> &input,
                            dax::cont::ArrayHandle<T,DeviceAdapterOpenMP>& output)
    {
    DAX_ASSERT_CONT(input.hasExecutionArray());
    DAX_ASSERT_CONT(output.GetNumberOfEntries() == input.GetNumberOfEntries());
    output.ReadyAsOutput();
    return dax::openmp::cont::inclusiveScan(input.GetExecutionArray(),
                                            output.GetExecutionArray());
    }

  template<typename T, typename U>
  static void LowerBounds(const dax::cont::ArrayHandle<T,DeviceAdapterOpenMP>& input,
                         const dax::cont::ArrayHandle<T,DeviceAdapterOpenMP>& values,
                         dax::cont::ArrayHandle<U,DeviceAdapterOpenMP>& output)
    {
    DAX_ASSERT_CONT(input.hasExecutionArray());
    DAX_ASSERT_CONT(values.hasExecutionArray());
    DAX_ASSERT_CONT(values.GetNumberOfEntries() <= output.GetNumberOfEntries());
    output.ReadyAsOutput();
    dax::openmp::cont::lowerBounds(input.GetExecutionArray(),
                                   values.GetExecutionArray(),
                                   output.GetExecutionArray());
    }

  template<class Functor, class Parameters>
  static void Schedule(Functor functor,
                       Parameters parameters,
                       dax::Id numInstances)
  {
    dax::openmp::cont::scheduleThrust(functor, parameters, numInstances);
  }

  template<typename T>
  static void Sort(dax::cont::ArrayHandle<T,DeviceAdapterOpenMP>& values)
    {
    DAX_ASSERT_CONT(values.hasExecutionArray());
    dax::openmp::cont::sort(values.GetExecutionArray());
    }

  template<typename T,typename U>
  static void StreamCompact(
      const dax::cont::ArrayHandle<T,DeviceAdapterOpenMP>& input,
      dax::cont::ArrayHandle<U,DeviceAdapterOpenMP>& output)
    {
    //the input array is both the input and the stencil output for the scan
    //step. In this case the index position is the input and the value at
    //each index is the stencil value
    DAX_ASSERT_CONT(input.hasExecutionArray());
    dax::openmp::cont::streamCompact(input.GetExecutionArray(),
                                  output.GetExecutionArray());
    output.UpdateArraySize();
    }

  template<typename T, typename U>
  static void StreamCompact(
      const dax::cont::ArrayHandle<T,DeviceAdapterOpenMP>& input,
      const dax::cont::ArrayHandle<U,DeviceAdapterOpenMP>& stencil,
      dax::cont::ArrayHandle<T,DeviceAdapterOpenMP>& output)
    {
    //the input array is both the input and the stencil output for the scan
    //step. In this case the index position is the input and the value at
    //each index is the stencil value
    DAX_ASSERT_CONT(input.hasExecutionArray());
    DAX_ASSERT_CONT(stencil.hasExecutionArray());
    dax::openmp::cont::streamCompact(input.GetExecutionArray(),
                                     stencil.GetExecutionArray(),
                                     output.GetExecutionArray());
    output.UpdateArraySize();
    }

  template<typename T>
  static void Unique(dax::cont::ArrayHandle<T,DeviceAdapterOpenMP>& values)
    {
    DAX_ASSERT_CONT(values.hasExecutionArray());
    dax::openmp::cont::unique(values.GetExecutionArray());
    values.UpdateArraySize(); //unique might resize the execution array
    }
};

}
}
} // namespace dax::openmp::cont

#endif //__dax_openmp_cont_DeviceAdapterOpenMP_h
