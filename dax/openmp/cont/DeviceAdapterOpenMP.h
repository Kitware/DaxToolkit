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
#include <dax/openmp/cont/StreamCompact.h>
#include <dax/openmp/cont/internal/ArrayContainerExecutionThrust.h>

namespace dax {
namespace cont {
  //forward declare the ArrayHandle before we use it.
template< typename OtherT, class OtherDeviceAdapter > class ArrayHandle;
}
}

#include <dax/cont/internal/ExecutionPackageGrid.h>

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

  template<typename T, class IteratorType>
  static void SetControlArray(
      dax::cont::ArrayHandle<T,DeviceAdapterOpenMP>& input,
      IteratorType begin, IteratorType end)
    {
      input.SetNewControlArray(begin,end);
    }

  template<class Functor, class Parameters>
  static void Schedule(Functor functor,
                       Parameters parameters,
                       dax::Id numInstances)
  {
    dax::openmp::cont::scheduleThrust(functor, parameters, numInstances);
  }


  template<typename T,typename U>
  static void StreamCompact(
      const dax::cont::ArrayHandle<T,DeviceAdapterOpenMP>& input,
      dax::cont::ArrayHandle<U,DeviceAdapterOpenMP>& output)
    {
    //the input array is both the input and the stencil output for the scan
    //step. In this case the index position is the input and the value at
    //each index is the stencil value
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
    dax::openmp::cont::streamCompact(input.GetExecutionArray(),
                                     stencil.GetExecutionArray(),
                                     output.GetExecutionArray());
    output.UpdateArraySize();
    }

  template<typename T,typename U>
  static void GenerateTopology(
      const dax::cont::internal::ExecutionPackageGrid<T> &inputGrid,
      dax::cont::internal::ExecutionPackageGrid<U> &outGrid,
      const dax::cont::ArrayHandle<dax::Id,DeviceAdapterOpenMP> &cellIdsToUse,
      const dax::cont::ArrayHandle<dax::Id,DeviceAdapterOpenMP> &pointIdsToUse)
    {

    }
};

}
}
} // namespace dax::openmp::cont

#endif //__dax_openmp_cont_DeviceAdapterOpenMP_h
