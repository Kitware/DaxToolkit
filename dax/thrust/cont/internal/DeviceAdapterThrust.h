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
#include <dax/thrust/cont/internal/DeviceAdapterThrustTag.h>

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

// These have to be in the dax::cont::internal namespace to match those
// defined elsewhere.
namespace dax {
namespace cont {
namespace internal {

namespace detail {

template<typename T, class Container, class Adapter>
std::pair<typename dax::cont::internal::
            ArrayManagerExecution<T,Container,Adapter>::ThrustIteratorConstType,
          typename dax::cont::internal::
            ArrayManagerExecution<T,Container,Adapter>::ThrustIteratorConstType>
PrepareForInput(const dax::cont::ArrayHandle<T,Container,Adapter> &array)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>
      Manager;
  typedef typename Manager::PortalConstType PortalConstType;
  typedef typename Manager::ThrustIteratorConstType ThrustIteratorConstType;

  PortalConstType portal = array.PrepareForInput();
  return std::make_pair(Manager::ThrustIteratorBegin(portal),
                        Manager::ThrustIteratorEnd(portal));
}

template<typename T, class Container, class Adapter>
std::pair<typename dax::cont::internal::
            ArrayManagerExecution<T,Container,Adapter>::ThrustIteratorType,
          typename dax::cont::internal::
            ArrayManagerExecution<T,Container,Adapter>::ThrustIteratorType>
PrepareForOutput(dax::cont::ArrayHandle<T,Container,Adapter> &array,
                 dax::Id numberOfValues)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>
      Manager;
  typedef typename Manager::PortalType PortalType;
  typedef typename Manager::ThrustIteratorType ThrustIteratorType;

  PortalType portal = array.PrepareForOutput(numberOfValues);
  return std::make_pair(Manager::ThrustIteratorBegin(portal),
                        Manager::ThrustIteratorEnd(portal));
}

template<typename T, class Container, class Adapter>
std::pair<typename dax::cont::internal::
            ArrayManagerExecution<T,Container,Adapter>::ThrustIteratorType,
          typename dax::cont::internal::
            ArrayManagerExecution<T,Container,Adapter>::ThrustIteratorType>
PrepareForInPlace(dax::cont::ArrayHandle<T,Container,Adapter> &array)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>
      Manager;
  typedef typename Manager::PortalType PortalType;
  typedef typename Manager::ThrustIteratorType ThrustIteratorType;

  PortalType portal = array.PrepareForInPlace();
  return std::make_pair(Manager::ThrustIteratorBegin(portal),
                        Manager::ThrustIteratorEnd(portal));
}

} // namespace detail

template<typename T, class CIn, class COut, class Adapter>
DAX_CONT_EXPORT void Copy(
    const dax::cont::ArrayHandle<T,CIn,Adapter> &input,
    dax::cont::ArrayHandle<T,COut,Adapter> &output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef typename dax::cont::internal::ArrayManagerExecution<T,CIn,Adapter>::
      ThrustIteratorConstType ThrustIteratorInType;
  typedef typename dax::cont::internal::ArrayManagerExecution<T,COut,Adapter>::
      ThrustIteratorType ThrustIteratorOutType;

  dax::Id numberOfValues = input.GetNumberOfValues();
  std::pair<ThrustIteratorInType, ThrustIteratorInType> inputIter =
      detail::PrepareForInput(input);
  std::pair<ThrustIteratorOutType, ThrustIteratorOutType> outputIter =
      detail::PrepareForOutput(output, numberOfValues);

  ::thrust::copy(inputIter.first, inputIter.second, outputIter.first);
}

template<typename T, class CIn, class COut, class Adapter>
DAX_CONT_EXPORT T InclusiveScan(
    const dax::cont::ArrayHandle<T,CIn,Adapter> &input,
    dax::cont::ArrayHandle<T,COut,Adapter>& output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef typename dax::cont::internal::ArrayManagerExecution<T,CIn,Adapter>::
      ThrustIteratorConstType ThrustIteratorInType;
  typedef typename dax::cont::internal::ArrayManagerExecution<T,COut,Adapter>::
      ThrustIteratorType ThrustIteratorOutType;

  dax::Id numberOfValues = input.GetNumberOfValues();

  std::pair<ThrustIteratorInType, ThrustIteratorInType> inputIter =
      detail::PrepareForInput(input);
  std::pair<ThrustIteratorOutType, ThrustIteratorOutType> outputIter =
      detail::PrepareForOutput(output, numberOfValues);

  if (numberOfValues <= 0) { return 0; }

  ThrustIteratorOutType result = ::thrust::inclusive_scan(inputIter.first,
                                                          inputIter.second,
                                                          outputIter.first);

  //return the value at the last index in the array, as that is the size
  return *(result - 1);
}

template<typename T, class CIn, class CVal, class COut, class Adapter>
DAX_CONT_EXPORT void LowerBounds(
    const dax::cont::ArrayHandle<T,CIn,Adapter>& input,
    const dax::cont::ArrayHandle<T,CVal,Adapter>& values,
    dax::cont::ArrayHandle<dax::Id,COut,Adapter>& output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef typename dax::cont::internal::ArrayManagerExecution<T,CIn,Adapter>::
      ThrustIteratorConstType ThrustIteratorInType;
  typedef typename dax::cont::internal::ArrayManagerExecution<T,CVal,Adapter>::
      ThrustIteratorConstType ThrustIteratorValType;
  typedef typename dax::cont::internal::ArrayManagerExecution<
      dax::Id,COut,Adapter>::ThrustIteratorType ThrustIteratorOutType;

  dax::Id numberOfValues = values.GetNumberOfValues();

  std::pair<ThrustIteratorInType, ThrustIteratorInType> inputIter =
      detail::PrepareForInput(input);
  std::pair<ThrustIteratorValType, ThrustIteratorValType> valuesIter =
      detail::PrepareForInput(values);
  std::pair<ThrustIteratorOutType, ThrustIteratorOutType> outputIter =
      detail::PrepareForOutput(output, numberOfValues);

  ::thrust::lower_bound(inputIter.first,
                        inputIter.second,
                        valuesIter.first,
                        valuesIter.second,
                        outputIter.first);
}

template<class CIn, class COut, class Adapter>
DAX_CONT_EXPORT void LowerBounds(
    const dax::cont::ArrayHandle<dax::Id,CIn,Adapter> &input,
    dax::cont::ArrayHandle<dax::Id,COut,Adapter> &values_output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef typename dax::cont::internal::ArrayManagerExecution<
      dax::Id,CIn,Adapter>::ThrustIteratorConstType ThrustIteratorInType;
  typedef typename dax::cont::internal::ArrayManagerExecution<
      dax::Id,COut,Adapter>::ThrustIteratorType ThrustIteratorOutType;

  std::pair<ThrustIteratorInType, ThrustIteratorInType> inputIter =
      detail::PrepareForInput(input);
  std::pair<ThrustIteratorOutType, ThrustIteratorOutType> outputIter =
      detail::PrepareForInPlace(values_output);

  ::thrust::lower_bound(inputIter.first,
                        inputIter.second,
                        outputIter.first,
                        outputIter.second,
                        outputIter.first);
}

namespace detail {

template<class FunctorType>
class ScheduleKernelThrust
{
public:
  DAX_CONT_EXPORT ScheduleKernelThrust(const FunctorType &functor)
    : Functor(functor)
  {  }

  DAX_EXEC_EXPORT void operator()(dax::Id index) const {
    this->Functor(index);
  }

private:
  FunctorType Functor;
};

} // namespace detail

template<class Functor>
DAX_CONT_EXPORT void Schedule(
    Functor functor,
    dax::Id numInstances,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  const dax::Id ERROR_ARRAY_SIZE = 1024;
  ::thrust::device_vector<char> errorArray(ERROR_ARRAY_SIZE);
  errorArray[0] = '\0';
  dax::exec::internal::ErrorMessageBuffer errorMessage(
        ::thrust::raw_pointer_cast(&(*errorArray.begin())),
        errorArray.size());

  functor.SetErrorMessageBuffer(errorMessage);

  detail::ScheduleKernelThrust<Functor> kernel(functor);

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

template<typename T, class Container, class Adapter>
DAX_CONT_EXPORT void Sort(
    dax::cont::ArrayHandle<T,Container,Adapter>& values,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef typename dax::cont::internal::ArrayManagerExecution<
      T,Container,Adapter>::ThrustIteratorType ThrustIteratorType;

  std::pair<ThrustIteratorType, ThrustIteratorType> iterators =
      detail::PrepareForInPlace(values);
  ::thrust::sort(iterators.first, iterators.second);
}


namespace detail {

template<class ValueIterator,
         typename T,
         typename U,
         class CStencil,
         class COut,
         class Adapter>
DAX_CONT_EXPORT void RemoveIf(
    ValueIterator valuesBegin,
    ValueIterator valuesEnd,
    const dax::cont::ArrayHandle<T,CStencil,Adapter>& stencil,
    dax::cont::ArrayHandle<U,COut,Adapter>& output)
{
  typedef typename dax::cont::internal::ArrayManagerExecution<
      T,CStencil,Adapter>::ThrustIteratorConstType ThrustIteratorStencilType;
  typedef typename dax::cont::internal::ArrayManagerExecution<
      U,COut,Adapter>::ThrustIteratorType ThrustIteratorOutType;

  std::pair<ThrustIteratorStencilType, ThrustIteratorStencilType> stencilIter =
      detail::PrepareForInput(stencil);

  dax::Id numLeft = ::thrust::count_if(stencilIter.first,
                                       stencilIter.second,
                                       dax::not_default_constructor<T>());

  std::pair<ThrustIteratorOutType, ThrustIteratorOutType> outputIter =
      detail::PrepareForOutput(output, numLeft);

  ::thrust::copy_if(valuesBegin,
                    valuesEnd,
                    stencilIter.first,
                    outputIter.first,
                    dax::not_default_constructor<T>());
}

} // namespace detail

template<typename T, class CStencil, class COut, class Adapter>
DAX_CONT_EXPORT void StreamCompact(
    const dax::cont::ArrayHandle<T,CStencil,Adapter>& stencil,
    dax::cont::ArrayHandle<dax::Id,COut,Adapter>& output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  dax::Id stencilSize = stencil.GetNumberOfValues();

  detail::RemoveIf(::thrust::make_counting_iterator<dax::Id>(0),
                   ::thrust::make_counting_iterator<dax::Id>(stencilSize),
                   stencil,
                   output);
}

template<typename T,
         typename U,
         class CIn,
         class CStencil,
         class COut,
         class Adapter>
DAX_CONT_EXPORT void StreamCompact(
    const dax::cont::ArrayHandle<U,CIn,Adapter>& input,
    const dax::cont::ArrayHandle<T,CStencil,Adapter>& stencil,
    dax::cont::ArrayHandle<U,COut,Adapter>& output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef typename dax::cont::internal::ArrayManagerExecution<
      U,CIn,Adapter>::ThrustIteratorConstType ThrustIteratorInType;

  std::pair<ThrustIteratorInType, ThrustIteratorInType> inputIter =
      detail::PrepareForInput(input);

  detail::RemoveIf(inputIter.first, inputIter.second, stencil, output);
}

template<typename T, class Container, class Adapter>
DAX_CONT_EXPORT void Unique(
    dax::cont::ArrayHandle<T,Container,Adapter> &values,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef typename dax::cont::internal::ArrayManagerExecution<
      T,Container,Adapter>::ThrustIteratorType ThrustIteratorType;

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
