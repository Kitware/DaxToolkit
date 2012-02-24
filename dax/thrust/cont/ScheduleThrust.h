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

template<class FunctorType, class ParametersType, class PassBothIds = void>
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

  //needed when calling schedule for a subset
  template<typename Tuple>
  DAX_EXEC_EXPORT void operator()(Tuple t) {
    this->Functor(this->Parameters, ::thrust::get<0>(t),
                  this->ErrorHandler);
  }
  //needed for when calling from schedule on a range
  DAX_EXEC_EXPORT void operator()(dax::Id t) {
    this->Functor(this->Parameters, t,this->ErrorHandler);
  }

private:
  FunctorType Functor;
  ParametersType Parameters;
  dax::exec::internal::ErrorHandler ErrorHandler;
};

//If the typedef REQUIRES_BOTH_IDS exists, it means
//that the functor needs to know the index in the for loop and the
//value at the index in the data array. This allows some functors to do
//proper mappings
template<class FunctorType, class ParametersType>
class ScheduleThrustKernel<FunctorType,ParametersType,
      typename boost::enable_if_c<FunctorType::REQUIRES_BOTH_IDS >::type>
{
public:
  ScheduleThrustKernel(
      const FunctorType &functor,
      const ParametersType &parameters,
      dax::thrust::cont::internal::ArrayContainerExecutionThrust<char> &errorArray)
    : Functor(functor),
      Parameters(parameters),
      ErrorHandler(errorArray.GetExecutionArray()) {}

  template<typename Tuple>
  DAX_EXEC_EXPORT void operator()(Tuple t) {
    this->Functor(this->Parameters, ::thrust::get<0>(t), ::thrust::get<1>(t),
                  this->ErrorHandler);
  }
  //needed for when calling from schedule on a range
  DAX_EXEC_EXPORT void operator()(dax::Id t) {
      this->Functor(this->Parameters,t,t,this->ErrorHandler);
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

template<class Functor, class Parameters>
DAX_CONT_EXPORT void scheduleThrust(Functor functor,
                                    Parameters parameters,
                                    dax::internal::DataArray<dax::Id> ids)
{
  const dax::Id ERROR_ARRAY_SIZE = 1024;
  dax::thrust::cont::internal::ArrayContainerExecutionThrust<char> errorArray;
  errorArray.Allocate(ERROR_ARRAY_SIZE);
  errorArray[0] = '\0';

  dax::thrust::exec::internal::kernel::ScheduleThrustKernel<Functor, Parameters>
      kernel(functor, parameters, errorArray);

  //we package up the real ids with the index values
  //so that functors that need both will have it
  ::thrust::device_ptr<dax::Id> dev_ptr_start =
      ::thrust::device_pointer_cast(ids.GetPointer());
  ::thrust::device_ptr<dax::Id> dev_ptr_end =
      dev_ptr_start + ids.GetNumberOfEntries();

  ::thrust::for_each(
    ::thrust::make_zip_iterator(::thrust::make_tuple(dev_ptr_start,
                                          ::thrust::make_counting_iterator(0))),
    ::thrust::make_zip_iterator(
        ::thrust::make_tuple(dev_ptr_end,::thrust::make_counting_iterator(
                                          ids.GetNumberOfEntries()))),
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
