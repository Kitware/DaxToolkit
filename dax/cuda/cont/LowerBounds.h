/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cuda_cont_LowerBounds_h
#define __dax_cuda_cont_LowerBounds_h

#include <dax/cuda/cont/internal/SetThrustForCuda.h>
#include <dax/thrust/cont/LowerBounds.h>

namespace dax {
namespace cuda {
namespace cont {


template<typename T, typename U>
DAX_CONT_EXPORT void lowerBounds(const T& input,
                                 const U& values,
                                 U& output)
{
  dax::thrust::cont::lowerBounds(input,values,output);
}

}
}
} // namespace dax::cuda::cont

#endif //__dax_cuda_cont_LowerBounds_h
