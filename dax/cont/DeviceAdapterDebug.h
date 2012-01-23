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

#include <dax/cont/ScheduleDebug.h>
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
  static char *Schedule(Functor functor,
                        Parameters parameters,
                        dax::Id numInstances)
  {
    return dax::cont::scheduleDebug(functor, parameters, numInstances);
  }

  template<typename T>
  class ArrayContainerExecution
      : public dax::cont::internal::ArrayContainerExecutionCPU<T> { };
};

}
} // namespace dax::cont

#endif //__dax_cont_DeviceAdapterDebug_h
