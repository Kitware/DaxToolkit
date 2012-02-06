/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cuda_cont_internal_ArrayContainerExecutionThrust_h
#define __dax_cuda_cont_internal_ArrayContainerExecutionThrust_h

#include <dax/cuda/cont/internal/SetThrustForCuda.h>

#include <dax/thrust/cont/internal/ArrayContainerExecutionThrust.h>

#include <dax/cont/ErrorControlOutOfMemory.h>

#include <cuda.h>

namespace dax {
namespace cuda {
namespace cont {
namespace internal {

/// Manages a CUDA device array. Can allocate the array of the given type on
/// the device, copy data do and from it, and release the memory. The memory is
/// also released when this object goes out of scope.
///
template<typename T>
class ArrayContainerExecutionThrust
    : public dax::thrust::cont::internal::ArrayContainerExecutionThrust<T>
{
public:
  typedef T ValueType;
  typedef dax::thrust::cont::internal::ArrayContainerExecutionThrust<T> Superclass;

  // Fixes problem encountered with thrust where if an out-of-memory error
  // is thrown, it leaves the CUDA error hanging around.
  void Allocate(dax::Id numEntries)
  {
    try
      {
      this->Superclass::Allocate(numEntries);
      }
    catch (...)
      {
      // Clear CUDA memory allocation error.
      cudaError cError = cudaPeekAtLastError();
      if (cError == cudaErrorMemoryAllocation)
        {
        cudaGetLastError();
        }
      // Continue to throw error.
      throw;
      }
  }
};

}
}
}
}

#endif // __dax_cuda_cont_internal_ArrayContainerExecutionThrust_h
