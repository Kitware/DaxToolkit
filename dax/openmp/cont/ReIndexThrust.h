/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_openmp_cont_ReIndexThrust_h
#define __dax_openmp_cont_ReIndexThrust_h

#include <dax/openmp/cont/internal/SetThrustForOpenMP.h>
#include <dax/thrust/cont/ReIndexThrust.h>

namespace dax {
namespace openmp {
namespace cont {

DAX_CONT_EXPORT void ReIndexThrust(dax::internal::DataArray<dax::Id> ids)
{
  dax::thrust::cont::ReIndexThrust(ids);
}

}
}
} // namespace dax::openmp::cont

#endif //__dax_openmp_cont_ReIndexThrust_h
