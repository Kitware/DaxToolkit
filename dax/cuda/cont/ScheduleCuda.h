/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cuda_cont_ScheduleCuda_h
#define __dax_cuda_cont_ScheduleCuda_h

#include <dax/Types.h>

#include <dax/cuda/cont/internal/ArrayContainerExecutionThrust.h>
#include <dax/cuda/cont/internal/CudaParameters.h>
#include <dax/cuda/exec/internal/CudaThreadIterator.h>
#include <dax/cont/ErrorExecution.h>
#include <dax/exec/internal/ErrorHandler.h>
#include <dax/internal/DataArray.h>

#include <thrust/copy.h>

namespace dax {
namespace cuda {
namespace exec {
namespace internal {
namespace kernel {

template<typename Functor, class Parameters>
__global__ void scheduleCudaKernel(Functor functor,
                                   Parameters parameters,
                                   dax::Id numInstances,
                                   dax::internal::DataArray<char> errorArray)
{
  dax::exec::internal::ErrorHandler errorHandler(errorArray);

  for (dax::cuda::exec::internal::CudaThreadIterator iter(numInstances);
       !iter.IsDone();
       iter.Next())
    {
    functor(parameters, iter.GetIndex(), errorHandler);
    }
}


template<typename Functor, class Parameters>
__global__ void scheduleCudaKernel(Functor functor,
                                   Parameters parameters,
                                   dax::internal::DataArray<dax::Id> instances,
                                   dax::Id numInstances,
                                   dax::internal::DataArray<char> errorArray)
{
  dax::exec::internal::ErrorHandler errorHandler(errorArray);

  for (dax::cuda::exec::internal::CudaThreadIterator iter(numInstances);
       !iter.IsDone();
       iter.Next())
    {
    functor(parameters, instances.GetValue(iter.GetIndex()), errorHandler);
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

namespace internal {

DAX_CONT_EXPORT
dax::cuda::cont::internal::ArrayContainerExecutionThrust<char> &
getScheduleCudaErrorArray()
{
  static dax::cuda::cont::internal::ArrayContainerExecutionThrust<char>
      ErrorArray;
  return ErrorArray;
}

}

template<class Functor, class Parameters>
DAX_CONT_EXPORT void scheduleCuda(Functor functor,
                                  Parameters parameters,
                                  dax::Id numInstances)
{
  const dax::Id ERROR_ARRAY_SIZE = 1024;
  dax::cuda::cont::internal::ArrayContainerExecutionThrust<char> &errorArray
      = internal::getScheduleCudaErrorArray();
  errorArray.Allocate(ERROR_ARRAY_SIZE);
  *errorArray.GetBeginThrustIterator() = '\0';

  dax::cuda::cont::internal::CudaParameters cudaParam(numInstances);
  dax::Id numBlocks = cudaParam.GetNumberOfBlocks();
  dax::Id numThreads = cudaParam.GetNumberOfThreads();

  using dax::cuda::exec::internal::kernel::scheduleCudaKernel;
  scheduleCudaKernel<<<numBlocks, numThreads>>>(functor,
                                                parameters,
                                                numInstances,
                                                errorArray.GetExecutionArray());

  if (*errorArray.GetBeginThrustIterator() != '\0')
    {
    char errorString[ERROR_ARRAY_SIZE];
    ::thrust::copy(errorArray.GetBeginThrustIterator(),
                   errorArray.GetEndThrustIterator(),
                   errorString);
    throw dax::cont::ErrorExecution(errorString);
    }
}

template<class Functor, class Parameters>
DAX_CONT_EXPORT void scheduleCuda(Functor functor,
                                  Parameters parameters,
                                  dax::internal::DataArray<dax::Id> instances)
{
  const dax::Id ERROR_ARRAY_SIZE = 1024;
  dax::cuda::cont::internal::ArrayContainerExecutionThrust<char> &errorArray
      = internal::getScheduleCudaErrorArray();
  errorArray.Allocate(ERROR_ARRAY_SIZE);
  *errorArray.GetBeginThrustIterator() = '\0';

  dax::Id numInstances = instances.GetNumberOfEntries();
  dax::cuda::cont::internal::CudaParameters cudaParam(numInstances);
  dax::Id numBlocks = cudaParam.GetNumberOfBlocks();
  dax::Id numThreads = cudaParam.GetNumberOfThreads();

  using dax::cuda::exec::internal::kernel::scheduleCudaKernel;
  scheduleCudaKernel<<<numBlocks, numThreads>>>(functor,
                                                parameters,
                                                instances,
                                                numInstances,
                                                errorArray.GetExecutionArray());

  if (*errorArray.GetBeginThrustIterator() != '\0')
    {
    char errorString[ERROR_ARRAY_SIZE];
    ::thrust::copy(errorArray.GetBeginThrustIterator(),
                   errorArray.GetEndThrustIterator(),
                   errorString);
    throw dax::cont::ErrorExecution(errorString);
    }

}

}
}
} // namespace dax::cuda::cont

#endif //__dax_cuda_cont_ScheduleCuda_h
