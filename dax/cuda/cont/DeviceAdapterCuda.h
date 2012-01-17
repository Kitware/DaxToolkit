/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_DeviceAdapterCuda_h
#define __dax_cont_DeviceAdapterCuda_h

#ifdef DAX_DEFAULT_DEVICE_ADAPTER
#undef DAX_DEFAULT_DEVICE_ADAPTER
#endif

#define DAX_DEFAULT_DEVICE_ADAPTER ::dax::cont::DeviceAdapterCuda

#include <dax/cuda/cont/ScheduleCuda.h>
#include <dax/cuda/cont/internal/ArrayContainerExecutionThrust.h>

namespace dax {
namespace cont {

/// An implementation of DeviceAdapter that will schedule execution on a GPU
/// using CUDA.
///
template<typename T = void>
struct DeviceAdapterCuda
{
  template<class Functor, class Parameters>
  static void Schedule(Functor functor,
                       Parameters parameters,
                       dax::Id numInstances)
  {
    dax::cuda::cont::scheduleCuda(functor, parameters, numInstances);
  }

  typedef dax::cuda::cont::internal::ArrayContainerExecutionThrust<T>
      ArrayContainerExecution;
};

}
} // namespace dax::cont

#endif //__dax_cont_DeviceAdapterCuda_h
