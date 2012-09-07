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
#ifndef __dax_cont_internal_DeviceAdapterAlgorithmSerial_h
#define __dax_cont_internal_DeviceAdapterAlgorithmSerial_h

#include <dax/Functional.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ErrorExecution.h>
#include <dax/cont/internal/DeviceAdapterAlgorithm.h>

#include <dax/exec/internal/ErrorMessageBuffer.h>
#include <dax/exec/internal/Functor.h>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/utility/enable_if.hpp>

#include <algorithm>
#include <numeric>

namespace dax {
namespace cont {
namespace internal {

template<typename T, class CIn, class COut>
DAX_CONT_EXPORT void Copy(
    const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTagSerial>& input,
    dax::cont::ArrayHandle<T,COut,DeviceAdapterTagSerial>& output,
    DeviceAdapterTagSerial)
{
  typedef typename dax::cont::ArrayHandle<T,COut,DeviceAdapterTagSerial>
      ::PortalExecution PortalOut;
  typedef typename dax::cont::ArrayHandle<T,CIn,DeviceAdapterTagSerial>
      ::PortalConstExecution PortalIn;

  dax::Id numberOfValues = input.GetNumberOfValues();
  PortalIn inputPortal = input.PrepareForInput();
  PortalOut outputPortal = output.PrepareForOutput(numberOfValues);

  std::copy(inputPortal.GetIteratorBegin(),
            inputPortal.GetIteratorEnd(),
            outputPortal.GetIteratorBegin());
}

template<typename T, class CIn, class COut>
DAX_CONT_EXPORT T InclusiveScan(
    const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTagSerial> &input,
    dax::cont::ArrayHandle<T,COut,DeviceAdapterTagSerial>& output,
    DeviceAdapterTagSerial)
{
  typedef typename dax::cont::ArrayHandle<T,COut,DeviceAdapterTagSerial>
      ::PortalExecution PortalOut;
  typedef typename dax::cont::ArrayHandle<T,CIn,DeviceAdapterTagSerial>
      ::PortalConstExecution PortalIn;

  dax::Id numberOfValues = input.GetNumberOfValues();

  PortalIn inputPortal = input.PrepareForInput();
  PortalOut outputPortal = output.PrepareForOutput(numberOfValues);

  if (numberOfValues <= 0) { return 0; }

  std::partial_sum(inputPortal.GetIteratorBegin(),
                   inputPortal.GetIteratorEnd(),
                   outputPortal.GetIteratorBegin());

  // Return the value at the last index in the array, which is the full sum.
  return outputPortal.Get(numberOfValues - 1);
}

template<typename T, class CIn, class CVal, class COut>
DAX_CONT_EXPORT void LowerBounds(
    const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTagSerial>& input,
    const dax::cont::ArrayHandle<T,CVal,DeviceAdapterTagSerial>& values,
    dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTagSerial>& output,
    DeviceAdapterTagSerial)
{
  typedef typename dax::cont::ArrayHandle<T,CIn,DeviceAdapterTagSerial>
      ::PortalConstExecution PortalIn;
  typedef typename dax::cont::ArrayHandle<T,CVal,DeviceAdapterTagSerial>
      ::PortalConstExecution PortalVal;
  typedef typename dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTagSerial>
      ::PortalExecution PortalOut;

  dax::Id numberOfValues = values.GetNumberOfValues();

  PortalIn inputPortal = input.PrepareForInput();
  PortalVal valuesPortal = values.PrepareForInput();
  PortalOut outputPortal = output.PrepareForOutput(numberOfValues);

  // std::lower_bound only supports a single value to search for so iterate
  // over all values and search for each one.
  for (dax::Id outputIndex = 0; outputIndex < numberOfValues; outputIndex++)
    {
    // std::lower_bound returns an iterator to the position where you can
    // insert, but we want the distance from the start.
    typename PortalIn::IteratorType resultPos =
        std::lower_bound(inputPortal.GetIteratorBegin(),
                         inputPortal.GetIteratorEnd(),
                         valuesPortal.Get(outputIndex));
    dax::Id resultIndex =
        static_cast<dax::Id>(std::distance(inputPortal.GetIteratorBegin(),
                                           resultPos));
    outputPortal.Set(outputIndex, resultIndex);
    }
}

template<class CIn, class COut>
DAX_CONT_EXPORT void LowerBounds(
    const dax::cont::ArrayHandle<dax::Id,CIn,DeviceAdapterTagSerial> &input,
    dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTagSerial> &values_output,
    DeviceAdapterTagSerial)
{
  typedef typename dax::cont::ArrayHandle<dax::Id,CIn,DeviceAdapterTagSerial>
      ::PortalConstExecution PortalIn;
  typedef typename dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTagSerial>
      ::PortalExecution PortalOut;

  PortalIn inputPortal = input.PrepareForInput();
  PortalOut outputPortal = values_output.PrepareForInPlace();

  dax::Id outputSize = outputPortal.GetNumberOfValues();
  for (dax::Id outputIndex = 0; outputIndex < outputSize; outputIndex++)
    {
    // std::lower_bound returns an iterator to the position where you can
    // insert, but we want the distance from the start.
    typename PortalIn::IteratorType resultPos =
        std::lower_bound(inputPortal.GetIteratorBegin(),
                         inputPortal.GetIteratorEnd(),
                         outputPortal.Get(outputIndex));
    dax::Id resultIndex =
        static_cast<dax::Id>(std::distance(inputPortal.GetIteratorBegin(),
                                           resultPos));
    outputPortal.Set(outputIndex, resultIndex);
    }
}

namespace detail {

// This runs in the execution environment.
template<class FunctorType>
class ScheduleKernelSerial
{
public:
  ScheduleKernelSerial(
      const FunctorType &functor)
    : Functor(functor) {  }

  //needed for when calling from schedule on a range
  DAX_EXEC_EXPORT void operator()(dax::Id index) const
  {
    this->Functor(index);
  }

private:
  const FunctorType Functor;
};

} // namespace detail

template<class Functor>
DAX_CONT_EXPORT void Schedule(Functor functor,
                              dax::Id numInstances,
                              DeviceAdapterTagSerial)
{
  const dax::Id MESSAGE_SIZE = 1024;
  char errorString[MESSAGE_SIZE];
  errorString[0] = '\0';
  dax::exec::internal::ErrorMessageBuffer
      errorMessage(errorString, MESSAGE_SIZE);

  functor.SetErrorMessageBuffer(errorMessage);

  detail::ScheduleKernelSerial<Functor> kernel(functor);

  std::for_each(
        ::boost::counting_iterator<dax::Id>(0),
        ::boost::counting_iterator<dax::Id>(numInstances),
        kernel);

  if (errorMessage.IsErrorRaised())
    {
    throw dax::cont::ErrorExecution(errorString);
    }
}

template<class ControlInvocSig, class Functor,  class Bindings>
DAX_CONT_EXPORT void NG_Schedule(
    Functor functor,
    Bindings bindings,
    dax::Id numInstances,
    DeviceAdapterTagSerial)
{
  const dax::Id MESSAGE_SIZE = 1024;
  char errorString[MESSAGE_SIZE];
  errorString[0] = '\0';
  dax::exec::internal::ErrorMessageBuffer
      errorMessage(errorString, MESSAGE_SIZE);

  functor.SetErrorMessageBuffer(errorMessage);

  //setup functor
  dax::exec::internal::Functor<ControlInvocSig> kernel(functor, bindings);
  std::for_each(
        ::boost::counting_iterator<dax::Id>(0),
        ::boost::counting_iterator<dax::Id>(numInstances),
        kernel);

  if (errorMessage.IsErrorRaised())
    {
    throw dax::cont::ErrorExecution(errorString);
    }
}


template<typename T, class Container>
DAX_CONT_EXPORT void Sort(
    dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>& values,
    DeviceAdapterTagSerial)
{
  typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>
      ::PortalExecution PortalType;

  PortalType arrayPortal = values.PrepareForInPlace();
  std::sort(arrayPortal.GetIteratorBegin(), arrayPortal.GetIteratorEnd());
}

template<typename T, class CStencil, class COut>
DAX_CONT_EXPORT void StreamCompact(
    const dax::cont::ArrayHandle<T,CStencil,DeviceAdapterTagSerial>& stencil,
    dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTagSerial>& output,
    DeviceAdapterTagSerial)
{
  typedef typename dax::cont::ArrayHandle<T,CStencil,DeviceAdapterTagSerial>
      ::PortalConstExecution PortalStencil;
  typedef typename dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTagSerial>
      ::PortalExecution PortalOut;

  PortalStencil stencilPortal = stencil.PrepareForInput();

  dax::Id outputSize = std::count_if(stencilPortal.GetIteratorBegin(),
                                     stencilPortal.GetIteratorEnd(),
                                     dax::not_default_constructor<dax::Id>());

  PortalOut outputPortal= output.PrepareForOutput(outputSize);

  dax::Id inputSize = stencilPortal.GetNumberOfValues();
  dax::Id outputIndex = 0;
  for (dax::Id inputIndex = 0; inputIndex < inputSize; inputIndex++)
    {
    // Only write index that does not match the default constructor of T
    T input = stencilPortal.Get(inputIndex);
    if (dax::not_default_constructor<T>()(input))
      {
      outputPortal.Set(outputIndex, inputIndex);
      outputIndex++;
      }
    }
  DAX_ASSERT_CONT(outputIndex == outputSize);
}

template<typename T, typename U, class CIn, class CStencil, class COut>
DAX_CONT_EXPORT void StreamCompact(
    const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTagSerial>& input,
    const dax::cont::ArrayHandle<U,CStencil,DeviceAdapterTagSerial>& stencil,
    dax::cont::ArrayHandle<T,COut,DeviceAdapterTagSerial>& output,
    DeviceAdapterTagSerial)
{
  typedef typename dax::cont::ArrayHandle<T,COut,DeviceAdapterTagSerial>
      ::PortalExecution PortalOut;
  typedef typename dax::cont::ArrayHandle<T,CIn,DeviceAdapterTagSerial>
      ::PortalConstExecution PortalIn;
  typedef typename dax::cont::ArrayHandle<U,CStencil,DeviceAdapterTagSerial>
      ::PortalConstExecution PortalStencil;

  PortalIn inputPortal = input.PrepareForInput();
  PortalStencil stencilPortal = stencil.PrepareForInput();

  dax::Id outputSize = std::count_if(stencilPortal.GetIteratorBegin(),
                                     stencilPortal.GetIteratorEnd(),
                                     dax::not_default_constructor<U>());

  PortalOut outputPortal = output.PrepareForOutput(outputSize);

  dax::Id inputSize = inputPortal.GetNumberOfValues();
  DAX_ASSERT_CONT(inputSize == stencilPortal.GetNumberOfValues());
  dax::Id outputIndex = 0;
  for (dax::Id inputIndex = 0; inputIndex < inputSize; inputIndex++)
    {
    U flag = stencilPortal.Get(inputIndex);
    // Only pass the input with a true flag in the stencil.
    if (dax::not_default_constructor<U>()(flag))
      {
      outputPortal.Set(outputIndex, inputPortal.Get(inputIndex));
      outputIndex++;
      }
    }
  DAX_ASSERT_CONT(outputIndex == outputSize);
}

template<typename T, class Container>
DAX_CONT_EXPORT void Unique(
    dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>& values,
    DeviceAdapterTagSerial)
{
  typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>
      ::PortalExecution PortalT;

  PortalT arrayPortal = values.PrepareForInPlace();

  typename PortalT::IteratorType newEnd =
      std::unique(arrayPortal.GetIteratorBegin(), arrayPortal.GetIteratorEnd());
  values.Shrink(std::distance(arrayPortal.GetIteratorBegin(), newEnd));
}

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_DeviceAdapterAlgorithmSerial_h
