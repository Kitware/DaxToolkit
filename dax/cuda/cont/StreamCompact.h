/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cuda_cont_StreamCompact_h
#define __dax_cuda_cont_StreamCompact_h

#include <dax/cuda/cont/internal/SetThrustForCuda.h>
#include <dax/thrust/cont/StreamCompact.h>

namespace dax {
namespace cuda {
namespace cont {

template<typename T, typename U>
DAX_CONT_EXPORT void streamCompact(const T& input, U& output)
{
  dax::thrust::cont::streamCompact(input,output);
}

template<typename T, typename U>
DAX_CONT_EXPORT void streamCompact(const T& input,
                                   const U& stencil,
                                   T& output)
{
  dax::thrust::cont::streamCompact(input,stencil,output);
}

template<typename T, typename U>
DAX_CONT_EXPORT void generateStencil(T& input,
                                   U& stencil)
{
  dax::thrust::cont::generateStencil(input,stencil);
}

}
}
} // namespace dax::cuda::cont

#endif //__dax_cuda_cont_StreamCompact_h
