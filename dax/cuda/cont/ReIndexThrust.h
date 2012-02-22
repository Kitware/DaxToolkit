/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cuda_cont_ReIndexThrust_h
#define __dax_cuda_cont_ReIndexThrust_h

#include <dax/cuda/cont/internal/SetThrustForCuda.h>
#include <dax/thrust/cont/ReIndexThrust.h>

namespace dax {
namespace cuda {
namespace cont {

DAX_CONT_EXPORT void ReIndexThrust(dax::internal::DataArray<dax::Id> ids)
{
  dax::thrust::cont::ReIndexThrust(ids);
}

}
}
} // namespace dax::cuda::cont

#endif //__dax_cuda_cont_ReIndexThrust_h
