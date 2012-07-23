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

#ifndef __dax_cont_DeviceAdapterSerial_h
#define __dax_cont_DeviceAdapterSerial_h

#ifdef DAX_DEFAULT_DEVICE_ADAPTER
#undef DAX_DEFAULT_DEVICE_ADAPTER
#endif

namespace dax {
namespace cont {

/// A simple implementation of a DeviceAdapter that can be used for debuging.
/// The scheduling will simply run everything in a serial loop, which is easy
/// to track in a debugger.
///
struct DeviceAdapterTagSerial {  };

}
}

#define DAX_DEFAULT_DEVICE_ADAPTER ::dax::cont::DeviceAdapterTagSerial

#include <dax/Functional.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/ErrorExecution.h>
#include <dax/cont/internal/ArrayManagerExecutionShareWithControl.h>

#include <dax/exec/internal/ErrorMessageBuffer.h>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/utility/enable_if.hpp>

#include <algorithm>
#include <numeric>

namespace dax {
namespace cont {
namespace internal {

template <typename T, class ArrayContainerControlTag>
class ArrayManagerExecution<T, ArrayContainerControlTag, DeviceAdapterTagSerial>
    : public dax::cont::internal::ArrayManagerExecutionShareWithControl
          <T, ArrayContainerControlTag>
{
public:
  typedef dax::cont::internal::ArrayManagerExecutionShareWithControl
      <T, ArrayContainerControlTag> Superclass;
  typedef typename Superclass::ValueType ValueType;
  typedef typename Superclass::PortalType PortalType;
  typedef typename Superclass::PortalConstType PortalConstType;
};

}
}
} // namespace dax::cont::internal

namespace dax {
namespace cont {
namespace internal {

template<typename T, class Container>
DAX_CONT_EXPORT void Copy(
    const dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>& input,
    dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>& output,
    DeviceAdapterTagSerial)
{
  typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>
      ::PortalExecution PortalType;
  typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>
      ::PortalConstExecution PortalConstType;

  dax::Id numberOfValues = input.GetNumberOfValues();
  PortalConstType inputPortal = input.PrepareForInput();
  PortalType outputPortal = output.PrepareForOutput(numberOfValues);

  std::copy(inputPortal.GetIteratorBegin(),
            inputPortal.GetIteratorEnd(),
            outputPortal.GetIteratorBegin());
}

template<typename T, class Container>
DAX_CONT_EXPORT T InclusiveScan(
    const dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial> &input,
    dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>& output,
    DeviceAdapterTagSerial)
{
  typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>
      ::PortalExecution PortalType;
  typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>
      ::PortalConstExecution PortalConstType;

  dax::Id numberOfValues = input.GetNumberOfValues();

  PortalConstType inputPortal = input.PrepareForInput();
  PortalType outputPortal = output.PrepareForOutput(numberOfValues);

  if (numberOfValues <= 0) { return 0; }

  std::partial_sum(inputPortal.GetIteratorBegin(),
                   inputPortal.GetIteratorEnd(),
                   outputPortal.GetIteratorBegin());

  // Return the value at the last index in the array, which is the full sum.
  return outputPortal.Get(numberOfValues - 1);
}

template<typename T, class Container>
DAX_CONT_EXPORT void LowerBounds(
    const dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>& input,
    const dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>& values,
    dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapterTagSerial>& output,
    DeviceAdapterTagSerial)
{
  typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>
      ::IteratorConstExecution IteratorConstT;
  typedef typename dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapterTagSerial>
      ::IteratorExecution IteratorId;

  dax::Id numberOfValues = values.GetNumberOfValues();

  std::pair<IteratorConstT, IteratorConstT> inputIter
      = input.PrepareForInput();
  std::pair<IteratorConstT, IteratorConstT> valuesIter
      = values.PrepareForInput();
  std::pair<IteratorId, IteratorId> outputIter =
      output.PrepareForOutput(numberOfValues);

  // std::lower_bound only supports a single value to search for so iterate
  // over all values and search for each one.
  IteratorConstT value;
  IteratorId out;
  for (value = valuesIter.first, out = outputIter.first;
       value != valuesIter.second;
       value++, out++)
    {
    // std::lower_bound returns an iterator to the position where you can
    // insert, but we want the distance from the start.
    IteratorConstT resultPos
        = std::lower_bound(inputIter.first, inputIter.second, *value);
    *out = static_cast<dax::Id>(std::distance(inputIter.first, resultPos));
    }
}

template<class Container>
DAX_CONT_EXPORT void LowerBounds(
    const dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapterTagSerial>
        &input,
    dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapterTagSerial>
        &values_output,
    DeviceAdapterTagSerial)
{
  typedef typename dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapterTagSerial>
      ::PortalConstExecution PortalConstT;
  typedef typename dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapterTagSerial>
      ::PortalExecution PortalId;

  PortalConstT inputPortal = input.PrepareForInput();
  PortalId outputPortal = values_output.PrepareForInPlace();

  dax::Id outputSize = outputPortal.GetNumberOfValues();
  for (dax::Id outputIndex = 0; outputIndex < outputSize; outputIndex++)
    {
    // std::lower_bound returns an iterator to the position where you can
    // insert, but we want the distance from the start.
    typename PortalConstT::IteratorType resultPos =
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
      const FunctorType &functor,
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
    : Functor(functor), ErrorMessage(errorMessage) {  }

  //needed for when calling from schedule on a range
  DAX_EXEC_EXPORT void operator()(dax::Id index)
  {
    this->Functor(index, this->ErrorMessage);
  }

private:
  FunctorType Functor;
  const dax::exec::internal::ErrorMessageBuffer &ErrorMessage;
};

} // namespace detail

template<class Functor>
DAX_CONT_EXPORT void Schedule(Functor functor,
                              dax::Id numInstances,
                              DeviceAdapterTagSerial)
{
  const dax::Id MESSAGE_SIZE = 1024;
  char *errorString[MESSAGE_SIZE];
  dax::exec::internal::ErrorMessageBuffer
      errorMessage(errorString, MESSAGE_SIZE);

  detail::ScheduleKernelSerial<Functor> kernel(functor, errorMessage);

  std::for_each(
        ::boost::counting_iterator<dax::Id>(0),
        ::boost::counting_iterator<dax::Id>(numInstances),
        kernel);
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

template<typename T, class Container>
DAX_CONT_EXPORT void StreamCompact(
    const dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>& stencil,
    dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapterTagSerial>& output,
    DeviceAdapterTagSerial)
{
  typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>
      ::PortalConstExecution PortalConstT;
  typedef typename dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapterTagSerial>
      ::PortalExecution PortalId;

  PortalConstT stencilPortal = stencil.PrepareForInput();

  dax::Id outputSize = std::count_if(stencilPortal.GetIteratorBegin(),
                                     stencilPortal.GetIteratorEnd(),
                                     dax::not_default_constructor<dax::Id>());

  PortalId outputPortal= output.PrepareForOutput(outputSize);

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

template<typename T, typename U, class Container>
DAX_CONT_EXPORT void StreamCompact(
    const dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>& input,
    const dax::cont::ArrayHandle<U,Container,DeviceAdapterTagSerial>& stencil,
    dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>& output,
    DeviceAdapterTagSerial)
{
  typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>
      ::PortalExecution PortalT;
  typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>
      ::PortalConstExecution PortalConstT;
  typedef typename dax::cont::ArrayHandle<U,Container,DeviceAdapterTagSerial>
      ::PortalConstExecution PortalConstU;

  PortalConstT inputPortal = input.PrepareForInput();
  PortalConstU stencilPortal = stencil.PrepareForInput();

  dax::Id outputSize = std::count_if(stencilPortal.GetIteratorBegin(),
                                     stencilPortal.GetIteratorEnd(),
                                     dax::not_default_constructor<U>());

  PortalT outputPortal = output.PrepareForOutput(outputSize);

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

#endif //__dax_cont_DeviceAdapterSerial_h
