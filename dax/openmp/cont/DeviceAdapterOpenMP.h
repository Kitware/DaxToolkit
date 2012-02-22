/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_openmp_cont_DeviceAdapterOpenMP_h
#define __dax_openmp_cont_DeviceAdapterOpenMP_h

#ifdef DAX_DEFAULT_DEVICE_ADAPTER
#undef DAX_DEFAULT_DEVICE_ADAPTER
#endif

#define DAX_DEFAULT_DEVICE_ADAPTER ::dax::openmp::cont::DeviceAdapterOpenMP

#include <dax/openmp/cont/ScheduleThrust.h>
#include <dax/openmp/cont/ReIndexThrust.h>
#include <dax/openmp/cont/internal/ArrayContainerExecutionThrust.h>

namespace dax {
namespace cont {
//forward declare the arrayhandle class
template<typename OtherT, typename OtherDevice> class ArrayHandle;
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
  template<class Functor, class Parameters>
  static void Schedule(Functor functor,
                       Parameters parameters,
                       dax::Id numInstances)
  {
    dax::openmp::cont::scheduleThrust(functor, parameters, numInstances);
  }

  template<class Functor, class Parameters, typename T>
  static void Schedule(Functor functor,
                       Parameters parameters,
                       dax::cont::ArrayHandle<T,DeviceAdapterOpenMP> &ids)
  {
    dax::openmp::cont::scheduleThrust(functor, parameters, ids.ReadyAsInput());
  }


  template<typename T>
  static void ReIndex(dax::cont::ArrayHandle<T,DeviceAdapterOpenMP>& ids)
  {
    dax::openmp::cont::ReIndexThrust(ids.ReadyAsInput());
  }

  template<typename T>
  class ArrayContainerExecution
      : public dax::openmp::cont::internal::ArrayContainerExecutionThrust<T>
  { };
};

}
}
} // namespace dax::openmp::cont

#endif //__dax_openmp_cont_DeviceAdapterOpenMP_h
