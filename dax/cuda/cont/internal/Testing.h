/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cuda_cont_internal_Testing_h
#define __dax_cuda_cont_internal_Testing_h

#include <dax/cont/internal/Testing.h>

#include <cuda.h>

namespace dax {
namespace cuda {
namespace cont {
namespace internal {

struct Testing
{
public:
  template<class Func>
  static DAX_CONT_EXPORT int Run(Func function)
  {
    int result = dax::cont::internal::Testing::Run(function);
    cudaError_t cudaError = cudaPeekAtLastError();
    if (cudaError != cudaSuccess)
      {
      std::cout << "***** Unchecked Cuda error." << std::endl
                << cudaGetErrorString(cudaError) << std::endl;
      return 1;
      }
    else
      {
      std::cout << "No Cuda error detected." << std::endl;
      }
    return result;
  }
};

}
}
}
} // namespace dax::cuda::cont::internal

#endif //__dax_cuda_cont_internal_Testing_h
