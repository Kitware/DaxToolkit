/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_thrust_cont_ScheduleThrust_h
#define __dax_thrust_cont_ScheduleThrust_h

#include <dax/Types.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

namespace dax {
namespace thrust {
namespace exec {
namespace internal {
namespace kernel {

template<class FunctorType, class ParametersType>
class ScheduleThrustKernel
{
public:
  ScheduleThrustKernel(const FunctorType &functor,
                        const ParametersType &parameters)
    : Functor(functor), Parameters(parameters) { }

  DAX_EXEC_EXPORT void operator()(dax::Id instance) {
    this->Functor(this->Parameters, instance);
  }

private:
  FunctorType Functor;
  ParametersType Parameters;
};

}
}
}
}
} // namespace dax::thrust::exec::internal::kernel

namespace dax {
namespace thrust {
namespace cont {

template<class Functor, class Parameters>
DAX_CONT_EXPORT void scheduleThrust(Functor functor,
                                    Parameters& parameters,
                                    dax::Id numInstances)
{
  dax::thrust::exec::internal::kernel::ScheduleThrustKernel<Functor, Parameters>
      kernel(functor, parameters);

  ::thrust::for_each(::thrust::make_counting_iterator<dax::Id>(0),
                     ::thrust::make_counting_iterator<dax::Id>(numInstances),
                     kernel);
}

}
}
} // namespace dax::thrust::cont

#endif //__dax_thrust_cont_ScheduleThrust_h
