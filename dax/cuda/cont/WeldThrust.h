/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cuda_cont_WeldThrust_h
#define __dax_cuda_cont_WeldThrust_h

#include <dax/cuda/cont/internal/SetThrustForCuda.h>
#include <dax/thrust/cont/WeldThrust.h>

namespace dax {
namespace cuda {
namespace cont {

DAX_CONT_EXPORT void WeldThrust(dax::internal::DataArray<dax::Id> ids)
{
  dax::thrust::cont::Weld(ids);
}

}
}
} // namespace dax::cuda::cont

#endif //__dax_cuda_cont_WeldThrust_h
