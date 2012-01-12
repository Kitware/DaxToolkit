/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_Schedule_h
#define __dax_cont_Schedule_h

// TODO: Come up with a better way to choose the appropriate implementation for
// scheduling. When that happens, this file will probably be obsolete.

#include <dax/Types.h>

#ifdef DAX_CUDA

#include <dax/cuda/cont/ScheduleCuda.h>

namespace dax {
namespace cont {

template<class Functor, class Parameters>
DAX_CONT_EXPORT void schedule(Functor functor,
                              Parameters parameters,
                              dax::Id numInstances)
{
  dax::cuda::cont::scheduleCuda(functor, parameters, numInstances);
}

}
}

#else // DAX_CUDA

#include <dax/cont/ScheduleDebug.h>

namespace dax {
namespace cont {

template<class Functor, class Parameters>
DAX_CONT_EXPORT void schedule(Functor functor,
                              Parameters parameters,
                              dax::Id numInstances)
{
  dax::cont::scheduleDebug(functor, parameters, numInstances);
}

}
}

#endif // DAX_CUDA

#endif //__dax_cont_Schedule_h
