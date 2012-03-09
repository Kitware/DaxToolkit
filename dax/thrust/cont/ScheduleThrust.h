/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_thrust_cont_ScheduleThrust_h
#define __dax_thrust_cont_ScheduleThrust_h

#include <dax/Types.h>

#include <dax/cont/ErrorExecution.h>
#include <dax/cont/internal/IteratorContainer.h>
#include <dax/exec/internal/ErrorHandler.h>
#include <dax/thrust/cont/internal/ArrayContainerExecutionThrust.h>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#include <boost/utility/enable_if.hpp>

namespace dax {
namespace thrust {
namespace exec {
namespace internal {
namespace kernel {

template<class FunctorType, class ParametersType>
class ScheduleThrustKernel
{
public:
  ScheduleThrustKernel(
      const FunctorType &functor,
      const ParametersType &parameters,
      dax::thrust::cont::internal::ArrayContainerExecutionThrust<char> &errorArray)
    : Functor(functor),
      Parameters(parameters),
      ErrorHandler(errorArray.GetExecutionArray()) {}

  DAX_EXEC_EXPORT void operator()(dax::Id index) {
    this->Functor(this->Parameters, index,this->ErrorHandler);
  }

private:
  FunctorType Functor;
  ParametersType Parameters;
  dax::exec::internal::ErrorHandler ErrorHandler;
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
                                    Parameters parameters,
                                    dax::Id numInstances)
{
  const dax::Id ERROR_ARRAY_SIZE = 1024;
  dax::thrust::cont::internal::ArrayContainerExecutionThrust<char> errorArray;
  errorArray.Allocate(ERROR_ARRAY_SIZE);
  errorArray[0] = '\0';

  dax::thrust::exec::internal::kernel::ScheduleThrustKernel<Functor, Parameters>
      kernel(functor, parameters, errorArray);

  ::thrust::for_each(::thrust::make_counting_iterator<dax::Id>(0),
                     ::thrust::make_counting_iterator<dax::Id>(numInstances),
                     kernel);

  if (errorArray[0] != '\0')
    {
    char errorString[ERROR_ARRAY_SIZE];
    errorArray.CopyFromExecutionToControl(
          dax::cont::internal::make_IteratorContainer(errorString,
                                                      ERROR_ARRAY_SIZE));
    throw dax::cont::ErrorExecution(errorString);
    }
}

}
}
} // namespace dax::thrust::cont

#endif //__dax_thrust_cont_ScheduleThrust_h
