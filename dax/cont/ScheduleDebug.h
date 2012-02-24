/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_Schedule_h
#define __dax_cont_Schedule_h

#include <dax/Types.h>
#include <dax/internal/DataArray.h>
#include <dax/cont/ErrorExecution.h>
#include <dax/exec/internal/ErrorHandler.h>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/utility/enable_if.hpp>

#include <map>
#include <algorithm>
#include <vector>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class FunctorType, class ParametersType, class PassBothIds = void>
class ScheduleKernel
{
public:
  ScheduleKernel(
      const FunctorType &functor,
      const ParametersType &parameters,
      dax::internal::DataArray<char> &errorArray)
    : Functor(functor),
      Parameters(parameters),
      ErrorArray(errorArray),
      ErrorHandler(errorArray) {}

  //needed when calling schedule for a subset
  template<typename Tuple>
  DAX_EXEC_EXPORT void operator()(Tuple t)
  {
    this->Functor(this->Parameters, boost::get<0>(t),
                  this->ErrorHandler);
    if (this->ErrorHandler.IsErrorRaised())
      {
      throw dax::cont::ErrorExecution(this->ErrorArray.GetPointer());
      }
  }
  //needed for when calling from schedule on a range
  DAX_EXEC_EXPORT void operator()(dax::Id t)
  {
    this->Functor(this->Parameters, t,this->ErrorHandler);
    if (this->ErrorHandler.IsErrorRaised())
      {
      throw dax::cont::ErrorExecution(this->ErrorArray.GetPointer());
      }
  }

private:
  FunctorType Functor;
  ParametersType Parameters;
  dax::internal::DataArray<char> ErrorArray;
  dax::exec::internal::ErrorHandler ErrorHandler;
};

//If the typedef REQUIRES_BOTH_IDS exists, it means
//that the functor needs to know the index in the for loop and the
//value at the index in the data array. This allows some functors to do
//proper mappings
template<class FunctorType, class ParametersType>
class ScheduleKernel<FunctorType,ParametersType,
      typename boost::enable_if_c<FunctorType::REQUIRES_BOTH_IDS >::type>
{
public:
  ScheduleKernel(
    const FunctorType &functor,
    const ParametersType &parameters,
    dax::internal::DataArray<char> &errorArray)
  : Functor(functor),
    Parameters(parameters),
    ErrorArray(errorArray),
    ErrorHandler(errorArray) {}

  template<typename Tuple>
  DAX_EXEC_EXPORT void operator()(Tuple t)
  {
    this->Functor(this->Parameters, boost::get<0>(t), boost::get<1>(t),
                  this->ErrorHandler);
    if (this->ErrorHandler.IsErrorRaised())
      {
      throw dax::cont::ErrorExecution(this->ErrorArray.GetPointer());
      }
  }

  //needed for when calling from schedule on a range
  DAX_EXEC_EXPORT void operator()(dax::Id t)
  {
    this->Functor(this->Parameters,t,t,this->ErrorHandler);
    if (this->ErrorHandler.IsErrorRaised())
      {
      throw dax::cont::ErrorExecution(this->ErrorArray.GetPointer());
      }
  }

private:
  FunctorType Functor;
  ParametersType Parameters;
  dax::internal::DataArray<char> ErrorArray;
  dax::exec::internal::ErrorHandler ErrorHandler;
};


}
}
}
} // namespace dax::exec::internal::kernel

namespace dax {
namespace cont {

namespace internal {

DAX_CONT_EXPORT dax::internal::DataArray<char> getScheduleDebugErrorArray()
{
  const dax::Id ERROR_ARRAY_SIZE = 1024;
  static char ErrorArrayBuffer[ERROR_ARRAY_SIZE];
  dax::internal::DataArray<char> ErrorArray(ErrorArrayBuffer, ERROR_ARRAY_SIZE);
  return ErrorArray;
}

}

template<class Functor, class Parameters>
DAX_CONT_EXPORT void scheduleDebug(Functor functor,
                                   Parameters parameters,
                                   dax::Id numInstances)
{
  dax::internal::DataArray<char> errorArray
      = internal::getScheduleDebugErrorArray();
  // Clear error value.
  errorArray.SetValue(0, '\0');

  dax::exec::internal::kernel::ScheduleKernel<Functor, Parameters>
      kernel(functor, parameters, errorArray);

  std::for_each(
        ::boost::counting_iterator<dax::Id>(0),
        ::boost::counting_iterator<dax::Id>(numInstances),
        kernel);
}

template<class Functor, class Parameters>
DAX_CONT_EXPORT void scheduleDebug(Functor functor,
                                   Parameters parameters,
                                   dax::internal::DataArray<dax::Id> ids)
{
  dax::internal::DataArray<char> errorArray
      = internal::getScheduleDebugErrorArray();
  // Clear error value.
  errorArray.SetValue(0, '\0');

  dax::exec::internal::kernel::ScheduleKernel<Functor, Parameters>
      kernel(functor, parameters, errorArray);

  dax::Id size = ids.GetNumberOfEntries();

  std::for_each(
        boost::make_zip_iterator(boost::make_tuple(ids.GetPointer(),
                                    ::boost::counting_iterator<dax::Id>(0))),
        boost::make_zip_iterator(boost::make_tuple(ids.GetPointer()+size,
                                    ::boost::counting_iterator<dax::Id>(size))),
        kernel);
}

}
} // namespace dax::cont

#endif //__dax_cont_Schedule_h
