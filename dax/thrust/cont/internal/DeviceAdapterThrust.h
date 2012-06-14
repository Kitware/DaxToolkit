//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_thrust_cont_internal_DeviceAdapterThrust_h
#define __dax_thrust_cont_internal_DeviceAdapterThrust_h

#include <dax/thrust/cont/internal/CheckThrustBackend.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ErrorExecution.h>

#include <dax/Functional.h>

#include <dax/exec/internal/ErrorMessageBuffer.h>

// Disable GCC warnings we check Dax for but Thrust does not.
#if defined(__GNUC__) && !defined(DAX_CUDA)
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#endif // gcc version >= 4.6
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 2)
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif // gcc version >= 4.2
#endif // gcc && !CUDA

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#if defined(__GNUC__) && !defined(DAX_CUDA)
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 6)
#pragma GCC diagnostic pop
#endif // gcc version >= 4.6
#endif // gcc && !CUDA

namespace dax {
namespace thrust {
namespace cont {
namespace internal {

/// This is an incomplete implementation of a device adapter. Specifically, it
/// is missing the ArrayManagerExecution and ExecutionAdapter classes and the
/// schedule function. This is why it is declared in an internal namespace.
/// However, all device adapters based on thrust should have their tags inherit
/// from this tag. This will make the implementation of the device adapter
/// algorithms automatically use those defined here.
///
/// The thrust namespace contains basic implementations of all missing parts.
/// The ArrayManagerExecution should be a trivial subclass of either
/// ArrayManagerExecutionThrustDevice or ArrayManagerExecutionThrustShare.
/// There is a basic type for ExecutionAdapter called ExecutionAdapterThrust
/// that thrust implementations should publicly subclass. Likewise, there is a
/// function called ScheduleThrust that implementations should trivially call
/// for their schedule implementation.
///
struct DeviceAdapterTagThrust {  };


template<class Container, class Adapter>
class ExecutionAdapterThrust
{
public:
  template <typename T>
  struct FieldStructures
  {
    typedef typename dax::cont::internal::ArrayManagerExecution
        <T,Container,Adapter>::IteratorType IteratorType;
    typedef typename dax::cont::internal::ArrayManagerExecution
        <T,Container,Adapter>::IteratorConstType IteratorConstType;
  };

  DAX_EXEC_EXPORT ExecutionAdapterThrust(char *messageBegin, char *messageEnd)
    : ErrorHandler(messageBegin, messageEnd) {  }

  DAX_EXEC_EXPORT void RaiseError(const char *message) const
  {
    this->ErrorHandler.RaiseError(message);
  }

private:
  dax::exec::internal::ErrorMessageBuffer<char *> ErrorHandler;
};


namespace detail {

template<class FunctorType,
         class ParametersType,
         class Container,
         class Adapter>
class ScheduleThrustKernel
{
public:
  DAX_CONT_EXPORT ScheduleThrustKernel(
      const FunctorType &functor,
      const ParametersType &parameters,
      ::thrust::device_vector<char> &errorMessage)
    : Functor(functor),
      Parameters(parameters),
      ErrorMessageBegin(::thrust::raw_pointer_cast(&(*errorMessage.begin()))),
      ErrorMessageEnd(::thrust::raw_pointer_cast(&(*errorMessage.end())))
  {  }

  DAX_EXEC_EXPORT void operator()(dax::Id index) {
    this->Functor(this->Parameters,
                  index,
                  dax::exec::internal::ExecutionAdapter<Container,Adapter>(
                    this->ErrorMessageBegin, this->ErrorMessageEnd));
  }

private:
  FunctorType Functor;
  ParametersType Parameters;
  char *ErrorMessageBegin;
  char *ErrorMessageEnd;
};

} // namespace detail

/// For technical reasons, we cannot template schedule for all thrust device
/// adapters and only for thrust device adapters. However, the implementation
/// is here, so thrust device adapter implementations just need to trivially
/// call this function.
///
template<class Functor,
         class Parameters,
         class Container,
         class Adapter>
DAX_CONT_EXPORT void ScheduleThrust(Functor functor,
                                    Parameters parameters,
                                    dax::Id numInstances,
                                    Container,
                                    Adapter)
{
  const dax::Id ERROR_ARRAY_SIZE = 1024;
  ::thrust::device_vector<char> errorArray(ERROR_ARRAY_SIZE);
  errorArray[0] = '\0';

  detail::ScheduleThrustKernel<Functor, Parameters, Container, Adapter> kernel(
        functor, parameters, errorArray);

  ::thrust::for_each(::thrust::make_counting_iterator<dax::Id>(0),
                     ::thrust::make_counting_iterator<dax::Id>(numInstances),
                     kernel);

  if (errorArray[0] != '\0')
    {
    char errorString[ERROR_ARRAY_SIZE];
    ::thrust::copy(errorArray.begin(), errorArray.end(), errorString);

    throw dax::cont::ErrorExecution(errorString);
    }
}

}
}
}
} // namespace dax::thrust::cont::internal

// These have to be in the dax::cont::internal namespace to match those
// defined elsewhere.
namespace dax {
namespace cont {
namespace internal {

namespace detail {

template<typename T, class Container, class Adapter>
std::pair<
typename dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>::ThrustIteratorConstType,
typename dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>::ThrustIteratorConstType>
PrepareForInput(const dax::cont::ArrayHandle<T,Container,Adapter> &array)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>
      Manager;
  typedef typename Manager::IteratorConstType IteratorConstType;
  typedef typename Manager::ThrustIteratorConstType ThrustIteratorConstType;

  std::pair<IteratorConstType, IteratorConstType> iterators =
      array.PrepareForInput();
  return std::make_pair(Manager::ThrustIterator(iterators.first),
                        Manager::ThrustIterator(iterators.second));
}

template<typename T, class Container, class Adapter>
std::pair<
typename dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>::ThrustIteratorType,
typename dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>::ThrustIteratorType>
PrepareForOutput(dax::cont::ArrayHandle<T,Container,Adapter> &array,
                 dax::Id numberOfValues)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>
      Manager;
  typedef typename Manager::IteratorType IteratorType;
  typedef typename Manager::ThrustIteratorType ThrustIteratorType;

  std::pair<IteratorType, IteratorType> iterators =
      array.PrepareForOutput(numberOfValues);
  return std::make_pair(Manager::ThrustIterator(iterators.first),
                        Manager::ThrustIterator(iterators.second));
}

template<typename T, class Container, class Adapter>
std::pair<
typename dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>::ThrustIteratorType,
typename dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>::ThrustIteratorType>
PrepareForInPlace(dax::cont::ArrayHandle<T,Container,Adapter> &array)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>
      Manager;
  typedef typename Manager::IteratorType IteratorType;
  typedef typename Manager::ThrustIteratorType ThrustIteratorType;

  std::pair<IteratorType, IteratorType> iterators =
      array.PrepareForInPlace();
  return std::make_pair(Manager::ThrustIterator(iterators.first),
                        Manager::ThrustIterator(iterators.second));
}

} // namespace detail

template<typename T, class Container, class Adapter>
DAX_CONT_EXPORT void Copy(
    const dax::cont::ArrayHandle<T,Container,Adapter> &from,
    dax::cont::ArrayHandle<T,Container,Adapter> &to,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>
      Manager;
  typedef typename Manager::ThrustIteratorType ThrustIteratorType;
  typedef typename Manager::ThrustIteratorConstType ThrustIteratorConstType;

  dax::Id numberOfValues = from.GetNumberOfValues();
  std::pair<ThrustIteratorConstType, ThrustIteratorConstType> fromIter =
      detail::PrepareForInput(from);
  std::pair<ThrustIteratorType, ThrustIteratorType> toIter =
      detail::PrepareForOutput(to, numberOfValues);

  ::thrust::copy(fromIter.first, fromIter.second, toIter.first);
}

template<typename T, class Container, class Adapter>
DAX_CONT_EXPORT T InclusiveScan(
    const dax::cont::ArrayHandle<T,Container,Adapter> &input,
    dax::cont::ArrayHandle<T,Container,Adapter>& output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>
      Manager;
  typedef typename Manager::ThrustIteratorType ThrustIteratorType;
  typedef typename Manager::ThrustIteratorConstType ThrustIteratorConstType;

  dax::Id numberOfValues = input.GetNumberOfValues();

  std::pair<ThrustIteratorConstType, ThrustIteratorConstType> inputIter =
      detail::PrepareForInput(input);
  std::pair<ThrustIteratorType, ThrustIteratorType> outputIter =
      detail::PrepareForOutput(output, numberOfValues);

  if (numberOfValues <= 0) { return 0; }

  ThrustIteratorType result = ::thrust::inclusive_scan(inputIter.first,
                                                       inputIter.second,
                                                       outputIter.first);

  //return the value at the last index in the array, as that is the size
  return *(result - 1);
}

template<typename T, class Container, class Adapter>
DAX_CONT_EXPORT void LowerBounds(
    const dax::cont::ArrayHandle<T,Container,Adapter>& input,
    const dax::cont::ArrayHandle<T,Container,Adapter>& values,
    dax::cont::ArrayHandle<dax::Id,Container,Adapter>& output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>
      ManagerT;
  typedef typename ManagerT::ThrustIteratorType ThrustIteratorT;
  typedef typename ManagerT::ThrustIteratorConstType ThrustIteratorConstT;
  typedef dax::cont::internal::ArrayManagerExecution<Id,Container,Adapter>
      ManagerId;
  typedef typename ManagerId::ThrustIteratorType ThrustIteratorId;
  typedef typename ManagerId::ThrustIteratorConstType ThrustIteratorConstId;

  dax::Id numberOfValues = values.GetNumberOfValues();

  std::pair<ThrustIteratorConstT, ThrustIteratorConstT> inputIter =
      detail::PrepareForInput(input);
  std::pair<ThrustIteratorConstT, ThrustIteratorConstT> valuesIter =
      detail::PrepareForInput(values);
  std::pair<ThrustIteratorId, ThrustIteratorId> outputIter =
      detail::PrepareForOutput(output, numberOfValues);

  ::thrust::lower_bound(inputIter.first,
                        inputIter.second,
                        valuesIter.first,
                        valuesIter.second,
                        outputIter.first);
}

template<class Container, class Adapter>
DAX_CONT_EXPORT void LowerBounds(
    const dax::cont::ArrayHandle<dax::Id,Container,Adapter> &input,
    dax::cont::ArrayHandle<dax::Id,Container,Adapter> &values_output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef dax::cont::internal::ArrayManagerExecution<Id,Container,Adapter>
      ManagerId;
  typedef typename ManagerId::ThrustIteratorType ThrustIteratorId;
  typedef typename ManagerId::ThrustIteratorConstType ThrustIteratorConstId;

  std::pair<ThrustIteratorConstId, ThrustIteratorConstId> inputIter =
      detail::PrepareForInput(input);
  std::pair<ThrustIteratorId, ThrustIteratorId> outputIter =
      detail::PrepareForInPlace(values_output);

  ::thrust::lower_bound(inputIter.first,
                        inputIter.second,
                        outputIter.first,
                        outputIter.second,
                        outputIter.first);
}

template<typename T, class Container, class Adapter>
DAX_CONT_EXPORT void Sort(
    dax::cont::ArrayHandle<T,Container,Adapter>& values,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>
      Manager;
  typedef typename Manager::ThrustIteratorType ThrustIteratorType;
  typedef typename Manager::ThrustIteratorConstType ThrustIteratorConstType;

  std::pair<ThrustIteratorType, ThrustIteratorType> iterators =
      detail::PrepareForInPlace(values);
  ::thrust::sort(iterators.first, iterators.second);
}


namespace detail {

template<class ValueIterator,
         typename T,
         typename U,
         class Container,
         class Adapter>
DAX_CONT_EXPORT void RemoveIf(
    ValueIterator valuesBegin,
    ValueIterator valuesEnd,
    const dax::cont::ArrayHandle<T,Container,Adapter>& stencil,
    dax::cont::ArrayHandle<U,Container,Adapter>& output)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>
      ManagerT;
  typedef typename ManagerT::ThrustIteratorType ThrustIteratorT;
  typedef typename ManagerT::ThrustIteratorConstType ThrustIteratorConstT;
  typedef dax::cont::internal::ArrayManagerExecution<U,Container,Adapter>
      ManagerU;
  typedef typename ManagerU::ThrustIteratorType ThrustIteratorU;
  typedef typename ManagerU::ThrustIteratorConstType ThrustIteratorConstU;

  std::pair<ThrustIteratorConstT, ThrustIteratorConstT> stencilIter =
      detail::PrepareForInput(stencil);

  dax::Id numLeft = ::thrust::count_if(stencilIter.first,
                                       stencilIter.second,
                                       dax::not_default_constructor<T>());

  std::pair<ThrustIteratorU, ThrustIteratorU> outputIter =
      detail::PrepareForOutput(output, numLeft);

  ::thrust::copy_if(valuesBegin,
                    valuesEnd,
                    stencilIter.first,
                    outputIter.first,
                    dax::not_default_constructor<T>());
}

} // namespace detail

template<typename T, class Container, class Adapter>
DAX_CONT_EXPORT void StreamCompact(
    const dax::cont::ArrayHandle<T,Container,Adapter>& stencil,
    dax::cont::ArrayHandle<dax::Id,Container,Adapter>& output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  dax::Id stencilSize = stencil.GetNumberOfValues();

  detail::RemoveIf(::thrust::make_counting_iterator<dax::Id>(0),
                   ::thrust::make_counting_iterator<dax::Id>(stencilSize),
                   stencil,
                   output);
}

template<typename T, typename U, class Container, class Adapter>
DAX_CONT_EXPORT void StreamCompact(
    const dax::cont::ArrayHandle<U,Container,Adapter>& input,
    const dax::cont::ArrayHandle<T,Container,Adapter>& stencil,
    dax::cont::ArrayHandle<U,Container,Adapter>& output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef dax::cont::internal::ArrayManagerExecution<U,Container,Adapter>
      ManagerU;
  typedef typename ManagerU::ThrustIteratorType ThrustIteratorU;
  typedef typename ManagerU::ThrustIteratorConstType ThrustIteratorConstU;

  std::pair<ThrustIteratorConstU, ThrustIteratorConstU> inputIter =
      detail::PrepareForInput(input);

  detail::RemoveIf(inputIter.first, inputIter.second, stencil, output);
}

template<typename T, class Container, class Adapter>
DAX_CONT_EXPORT void Unique(
    dax::cont::ArrayHandle<T,Container,Adapter> &values,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>
      Manager;
  typedef typename Manager::ThrustIteratorType ThrustIteratorType;
  typedef typename Manager::ThrustIteratorConstType ThrustIteratorConstType;

  std::pair<ThrustIteratorType, ThrustIteratorType> valueIter =
      detail::PrepareForInPlace(values);

  ThrustIteratorType newEnd =
      ::thrust::unique(valueIter.first, valueIter.second);

  values.Shrink(::thrust::distance(valueIter.first, newEnd));
}

}
}
} // namespace dax::cont::internal

#endif //__dax_thrust_cont_internal_DeviceAdapterThrust_h
