/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cuda_cont_StreamCompact_h
#define __dax_cuda_cont_StreamCompact_h

#include <dax/thrust/cont/StreamCompact.h>

namespace dax {
namespace cuda {
namespace cont {

template<typename T, typename U>
DAX_CONT_EXPORT void streamCompact(const T& t, U& u)
{
  dax::thrust::cont::streamCompact(t,u);
}

}
}
} // namespace dax::cuda::cont

#endif //__dax_cuda_cont_StreamCompact_h
