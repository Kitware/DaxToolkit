/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_openmp_cont_WeldThrust_h
#define __dax_openmp_cont_WeldThrust_h

#include <dax/openmp/cont/internal/SetThrustForOpenMP.h>
#include <dax/thrust/cont/WeldThrust.h>

namespace dax {
namespace openmp {
namespace cont {

DAX_CONT_EXPORT void WeldThrust(dax::internal::DataArray<dax::Id> ids)
{
  dax::thrust::cont::Weld(ids);
}

}
}
} // namespace dax::openmp::cont

#endif //__dax_openmp_cont_WeldThrust_h
