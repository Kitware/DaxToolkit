/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_openmp_cont_StreamCompact_h
#define __dax_openmp_cont_StreamCompact_h

#include <dax/openmp/cont/internal/SetThrustForOpenMP.h>
#include <dax/thrust/cont/StreamCompact.h>

namespace dax {
namespace openmp {
namespace cont {

template<typename T>
DAX_CONT_EXPORT void streamCompact(const T& input, T& output)
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

}
}
} // namespace dax::openmap::cont

#endif //__dax_openmp_cont_StreamCompact_h
