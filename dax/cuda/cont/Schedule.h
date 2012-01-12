/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cuda_cont_Schedule_h
#define __dax_cuda_cont_Schedule_h

#include <dax/Types.h>

#include <dax/cuda/cont/internal/CudaParameters.h>
#include <dax/cuda/exec/internal/CudaThreadIterator.h>

namespace dax {
namespace cuda {
namespace exec {
namespace internal {
namespace kernel {

template<typename Functor, class Parameters>
__global__ void scheduleKernel(Functor functor,
                               Parameters parameters,
                               dax::Id numInstances)
{
  for (dax::cuda::exec::internal::CudaThreadIterator iter(numInstances);
       !iter.IsDone();
       iter.Next())
    {
    functor(parameters, iter.GetIndex());
    }
}

}
}
}
}
} // namespace dax::cuda::exec::internal::kernel

namespace dax {
namespace cuda {
namespace cont {

template<class Functor, class Parameters>
DAX_CONT_EXPORT void schedule(Functor functor,
                              Parameters parameters,
                              dax::Id numInstances)
{
  dax::cuda::cont::internal::CudaParameters cudaParam(numInstances);
  dax::Id numBlocks = cudaParam.GetNumberOfBlocks();
  dax::Id numThreads = cudaParam.GetNumberOfThreads();

  using dax::cuda::exec::internal::kernel::scheduleKernel;
  scheduleKernel<<<numBlocks, numThreads>>>(functor, parameters, numInstances);
}

}
}
} // namespace dax::cuda::cont

#endif //__dax_cuda_cont_Schedule_h
