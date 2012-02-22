/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_openmp_cont_ScheduleThrust_h
#define __dax_openmp_cont_ScheduleThrust_h

#include <dax/openmp/cont/internal/SetThrustForOpenMP.h>

#include <dax/thrust/cont/ScheduleThrust.h>

namespace dax {
namespace openmp {
namespace cont {

template<class Functor, class Parameters>
DAX_CONT_EXPORT void scheduleThrust(Functor functor,
                                  Parameters parameters,
                                  dax::Id numInstances)
{
  dax::thrust::cont::scheduleThrust(functor, parameters, numInstances);
}

template<class Functor, class Parameters>
static void scheduleThrust(Functor functor,
                     Parameters parameters,
                     dax::internal::DataArray<dax::Id> ids)
{
  dax::thrust::cont::scheduleThrust(functor, parameters, ids);
}

}
}
} // namespace dax::cuda::cont

#endif //__dax_cuda_cont_ScheduleThrust_h
