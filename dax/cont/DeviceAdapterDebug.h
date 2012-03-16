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

#ifndef __dax_cont_DeviceAdapterDebug_h
#define __dax_cont_DeviceAdapterDebug_h

#ifdef DAX_DEFAULT_DEVICE_ADAPTER
#undef DAX_DEFAULT_DEVICE_ADAPTER
#endif

#define DAX_DEFAULT_DEVICE_ADAPTER ::dax::cont::DeviceAdapterDebug

#include <dax/internal/DataArray.h>
#include <dax/cont/CopyDebug.h>
#include <dax/cont/LowerBoundsDebug.h>
#include <dax/cont/ScheduleDebug.h>
#include <dax/cont/SortDebug.h>
#include <dax/cont/StreamCompactDebug.h>
#include <dax/cont/UniqueDebug.h>
#include <dax/cont/internal/ArrayContainerExecutionCPU.h>



namespace dax {
namespace cont {
  //forward declare the ArrayHandle before we use it.
  template< typename OtherT, class OtherDeviceAdapter > class ArrayHandle;
}
}

/// A simple implementation of a DeviceAdapter that can be used for debuging.
/// The scheduling will simply run everything in a serial loop, which is easy
/// to track in a debugger.
///
namespace dax {
namespace cont {
struct DeviceAdapterDebug
{
  template<typename T>
  class ArrayContainerExecution
      : public dax::cont::internal::ArrayContainerExecutionCPU<T> { };

  template<typename T>
  static void Copy(const dax::cont::ArrayHandle<T,DeviceAdapterDebug>& from,
                   dax::cont::ArrayHandle<T,DeviceAdapterDebug>& to)
    {
    DAX_ASSERT_CONT(from.hasExecutionArray());
    DAX_ASSERT_CONT(to.GetNumberOfEntries() >= from.GetNumberOfEntries());
    to.ReadyAsOutput();
    dax::cont::copyDebug(from.GetExecutionArray(),to.GetExecutionArray());
    }


  template<typename T, typename U>
  static void LowerBounds(const dax::cont::ArrayHandle<T,DeviceAdapterDebug>& input,
                         const dax::cont::ArrayHandle<T,DeviceAdapterDebug>& values,
                         dax::cont::ArrayHandle<U,DeviceAdapterDebug>& output)
    {
    DAX_ASSERT_CONT(input.hasExecutionArray());
    DAX_ASSERT_CONT(values.hasExecutionArray());
    DAX_ASSERT_CONT(output.hasExecutionArray());
    DAX_ASSERT_CONT(values.GetNumberOfEntries() <= output.GetNumberOfEntries());
    dax::cont::lowerBoundsDebug(input.GetExecutionArray(),
                                values.GetExecutionArray(),
                                output.GetExecutionArray());
    }

  template<class Functor, class Parameters>
  static void Schedule(Functor functor,
                       Parameters parameters,
                       dax::Id numInstances)
  {
    dax::cont::scheduleDebug(functor, parameters, numInstances);
  }


  template<typename T>
  static void Sort(dax::cont::ArrayHandle<T,DeviceAdapterDebug>& values)
    {
    DAX_ASSERT_CONT(values.hasExecutionArray());
    dax::cont::sortDebug(values.GetExecutionArray());
    }

  template<typename T,typename U>
  static void StreamCompact(
      const dax::cont::ArrayHandle<T,DeviceAdapterDebug>& input,
      dax::cont::ArrayHandle<U,DeviceAdapterDebug>& output)
    {
    //the input array is both the input and the stencil output for the scan
    //step. In this case the index position is the input and the value at
    //each index is the stencil value
    dax::cont::streamCompactDebug(input.GetExecutionArray(),
                                  output.GetExecutionArray());
    output.UpdateArraySize();
    }

  template<typename T, typename U>
  static void StreamCompact(
      const dax::cont::ArrayHandle<T,DeviceAdapterDebug>& input,
      const dax::cont::ArrayHandle<U,DeviceAdapterDebug>& stencil,
      dax::cont::ArrayHandle<T,DeviceAdapterDebug>& output)
    {
    //the input array is both the input and the stencil output for the scan
    //step. In this case the index position is the input and the value at
    //each index is the stencil value
    dax::cont::streamCompactDebug(input.GetExecutionArray(),
                                  stencil.GetExecutionArray(),
                                  output.GetExecutionArray());
    output.UpdateArraySize();
    }
  
  template<typename T>
  static void Unique(dax::cont::ArrayHandle<T,DeviceAdapterDebug>& values)
    {
    DAX_ASSERT_CONT(values.hasExecutionArray());
    dax::cont::uniqueDebug(values.GetExecutionArray());
    values.UpdateArraySize(); //unique might resize the execution array
    }

};

}
} // namespace dax::cont

#endif //__dax_cont_DeviceAdapterDebug_h
