/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cuda_cont_Sort_h
#define __dax_cuda_cont_Sort_h

#include <dax/cuda/cont/internal/SetThrustForCuda.h>
#include <dax/thrust/cont/Sort.h>

namespace dax {
namespace cuda {
namespace cont {


template<typename T>
DAX_CONT_EXPORT void sort(T &values)
{
  dax::thrust::cont::sort(values);
}

}
}
} // namespace dax::cuda::cont

#endif //__dax_cuda_cont_Sort_h
