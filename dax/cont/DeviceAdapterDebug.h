/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_DeviceAdapterDebug_h
#define __dax_cont_DeviceAdapterDebug_h

#ifdef DAX_DEFAULT_DEVICE_ADAPTER
#undef DAX_DEFAULT_DEVICE_ADAPTER
#endif

#define DAX_DEFAULT_DEVICE_ADAPTER ::dax::cont::DeviceAdapterDebug

#include <dax/internal/DataArray.h>
#include <dax/cont/ScheduleDebug.h>
#include <dax/cont/StreamCompactDebug.h>
#include <dax/cont/internal/ArrayContainerExecutionCPU.h>

namespace dax {
namespace cont {

/// A simple implementation of a DeviceAdapter that can be used for debuging.
/// The scheduling will simply run everything in a serial loop, which is easy
/// to track in a debugger.
///
struct DeviceAdapterDebug
{
  template<class Functor, class Parameters>
  static void Schedule(Functor functor,
                       Parameters parameters,
                       dax::Id numInstances)
  {
    dax::cont::scheduleDebug(functor, parameters, numInstances);
  }

  template<typename T>
  class ArrayContainerExecution
      : public dax::cont::internal::ArrayContainerExecutionCPU<T> { };

  template<typename HandleType>
  static HandleType StreamCompact(const HandleType& input)
    {
    //the input array is both the input and the stencil output for the scan
    //step. In this case the index position is the input and the value at
    //each index is the stencil value
    HandleType output;
//    dax::cont::streamCompactDebug(
//          dax::cont::internal::DeviceArray(input),
//          dax::cont::internal::DeviceArray(output)
//          );
//    dax::cont::internal::UpdateArrayHandleSize(output);
    return output;
    }
};

}
} // namespace dax::cont

#endif //__dax_cont_DeviceAdapterDebug_h
