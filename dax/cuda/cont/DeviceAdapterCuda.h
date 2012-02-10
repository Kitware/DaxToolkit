/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cuda_cont_DeviceAdapterCuda_h
#define __dax_cuda_cont_DeviceAdapterCuda_h

#ifdef DAX_DEFAULT_DEVICE_ADAPTER
#undef DAX_DEFAULT_DEVICE_ADAPTER
#endif
#define DAX_DEFAULT_DEVICE_ADAPTER ::dax::cuda::cont::DeviceAdapterCuda

//#define DAX_CUDA_NATIVE_SCHEDULE

#ifdef DAX_CUDA_NATIVE_SCHEDULE
#include <dax/cuda/cont/ScheduleCuda.h>
#else
#include <dax/cuda/cont/ScheduleThrust.h>
#endif

#include <dax/cuda/cont/StreamCompact.h>

#include <dax/cuda/cont/internal/ArrayContainerExecutionThrust.h>

namespace dax {
namespace cont {
  //forward declare the ArrayHandle before we use it.
template< typename OtherT, class OtherDeviceAdapter > class ArrayHandle;
}
}


namespace dax {
namespace cuda {
namespace cont {

/// An implementation of DeviceAdapter that will schedule execution on a GPU
/// using CUDA.
///
struct DeviceAdapterCuda
{
  template<typename T>
  class ArrayContainerExecution
      : public dax::cuda::cont::internal::ArrayContainerExecutionThrust<T>
  { };

  template<typename T, class IteratorType>
  static void SetControlArray(
      dax::cont::ArrayHandle<T,DeviceAdapterCuda>& input,
      IteratorType begin, IteratorType end)
    {
      input.SetNewControlArray(begin,end);
    }

  template<class Functor, class Parameters>
  static void Schedule(Functor functor,
                       Parameters parameters,
                       dax::Id numInstances)
  {
#ifdef DAX_CUDA_NATIVE_SCHEDULE
    dax::cuda::cont::scheduleCuda(functor, parameters, numInstances);
#else
    dax::cuda::cont::scheduleThrust(functor, parameters, numInstances);
#endif
  }

  template<typename T,typename U>
  static void StreamCompact(
      const dax::cont::ArrayHandle<T,DeviceAdapterCuda>& input,
      dax::cont::ArrayHandle<U,DeviceAdapterCuda>& output)
    {
    //the input array is both the input and the stencil output for the scan
    //step. In this case the index position is the input and the value at
    //each index is the stencil value
    dax::cuda::cont::streamCompact(input.GetExecutionArray(),
                                  output.GetExecutionArray());
    output.UpdateArraySize();
    }

  template<typename T,typename U>
  static void StreamCompact(
      const dax::cont::ArrayHandle<T,DeviceAdapterCuda>& input,
      const dax::cont::ArrayHandle<U,DeviceAdapterCuda>& stencil,
      dax::cont::ArrayHandle<T,DeviceAdapterCuda>& output)
    {
    //the input array is both the input and the stencil output for the scan
    //step. In this case the index position is the input and the value at
    //each index is the stencil value
    dax::cuda::cont::streamCompact(input.GetExecutionArray(),
                                   stencil.GetExecutionArray(),
                                   output.GetExecutionArray());
    output.UpdateArraySize();
    }
};

}
}
} // namespace dax::cuda::cont

#endif //__dax_cuda_cont_DeviceAdapterCuda_h
