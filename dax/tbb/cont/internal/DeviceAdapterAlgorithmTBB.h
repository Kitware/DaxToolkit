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
#ifndef __dax_tbb_cont_internal_DeviceAdapterAlgorithmTBB_h
#define __dax_tbb_cont_internal_DeviceAdapterAlgorithmTBB_h

#include <dax/tbb/cont/internal/DeviceAdapterTagTBB.h>
#include <dax/tbb/cont/internal/ArrayManagerExecutionTBB.h>

#include <dax/exec/internal/ErrorMessageBuffer.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ErrorExecution.h>
#include <dax/cont/internal/DeviceAdapterAlgorithmGeneral.h>

#include <boost/type_traits/remove_reference.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <tbb/parallel_sort.h>

namespace dax {
namespace cont {
namespace internal {

template<typename T, class CIn, class COut>
DAX_CONT_EXPORT void Copy(
    const dax::cont::ArrayHandle<T, CIn, dax::tbb::cont::DeviceAdapterTagTBB>
        &daxNotUsed(input),
    dax::cont::ArrayHandle<T, COut, dax::tbb::cont::DeviceAdapterTagTBB>
        &daxNotUsed(output),
    dax::tbb::cont::DeviceAdapterTagTBB)
{
  //TODO
}

namespace detail {

// The "grain size" of scheduling with TBB.  Not a lot of thought as gone
// into picking this size.
const dax::Id TBB_GRAIN_SIZE = 128;

template<class InputPortalType, class OutputPortalType>
struct InclusiveScanBodyTBB
{
  typedef typename boost::remove_reference<
      typename OutputPortalType::ValueType>::type ValueType;
  ValueType Sum;
  InputPortalType InputPortal;
  OutputPortalType OutputPortal;

  DAX_CONT_EXPORT
  InclusiveScanBodyTBB(const InputPortalType &inputPortal,
                       const OutputPortalType &outputPortal)
    : Sum(ValueType(0)), InputPortal(inputPortal), OutputPortal(outputPortal)
  {  }

  template<typename Tag>
  DAX_EXEC_EXPORT
  void operator()(const ::tbb::blocked_range<dax::Id> &range, Tag)
  {
    for (dax::Id index = range.begin(); index < range.end(); index++)
      {
      this->Sum = this->Sum + this->InputPortal.Get(index);
      if (Tag::is_final_scan())
        {
        this->OutputPortal.Set(index, this->Sum);
        }
      }
  }

  DAX_EXEC_CONT_EXPORT
  InclusiveScanBodyTBB(const InclusiveScanBodyTBB &body, ::tbb::split)
    : Sum(0),
      InputPortal(body.InputPortal),
      OutputPortal(body.OutputPortal) {  }

  DAX_EXEC_CONT_EXPORT
  void reverse_join(const InclusiveScanBodyTBB &left)
  {
    this->Sum = left.Sum + this->Sum;
  }

  DAX_EXEC_CONT_EXPORT
  void assign(const InclusiveScanBodyTBB &src)
  {
    this->Sum = src.Sum;
  }
};

template<class InputPortalType, class OutputPortalType>
typename boost::remove_reference<typename OutputPortalType::ValueType>::type
InclusiveScanPortals(InputPortalType inputPortal, OutputPortalType outputPortal)
{
  InclusiveScanBodyTBB<InputPortalType, OutputPortalType>
      body(inputPortal, outputPortal);
  dax::Id arrayLength = inputPortal.GetNumberOfValues();
  ::tbb::parallel_scan(
        ::tbb::blocked_range<dax::Id>(0, arrayLength, TBB_GRAIN_SIZE),
        body);
  return body.Sum;
}

template<class InputPortalType, class OutputPortalType>
struct ExclusiveScanBodyTBB
{
  typedef typename boost::remove_reference<
      typename OutputPortalType::ValueType>::type ValueType;
  ValueType Sum;
  InputPortalType InputPortal;
  OutputPortalType OutputPortal;

  DAX_CONT_EXPORT
  ExclusiveScanBodyTBB(const InputPortalType &inputPortal,
                       const OutputPortalType &outputPortal)
    : Sum(ValueType(0)), InputPortal(inputPortal), OutputPortal(outputPortal)
  {  }

  template<typename Tag>
  DAX_EXEC_EXPORT
  void operator()(const ::tbb::blocked_range<dax::Id> &range, Tag)
  {
    for (dax::Id index = range.begin(); index < range.end(); index++)
      {
      ValueType inputValue = this->InputPortal.Get(index);
      if (Tag::is_final_scan())
        {
        this->OutputPortal.Set(index, this->Sum);
        }
      this->Sum = this->Sum + inputValue;
      }
  }

  DAX_EXEC_CONT_EXPORT
  ExclusiveScanBodyTBB(const ExclusiveScanBodyTBB &body, ::tbb::split)
    : Sum(0),
      InputPortal(body.InputPortal),
      OutputPortal(body.OutputPortal) {  }

  DAX_EXEC_CONT_EXPORT
  void reverse_join(const ExclusiveScanBodyTBB &left)
  {
    this->Sum = left.Sum + this->Sum;
  }

  DAX_EXEC_CONT_EXPORT
  void assign(const ExclusiveScanBodyTBB &src)
  {
    this->Sum = src.Sum;
  }
};

template<class InputPortalType, class OutputPortalType>
typename boost::remove_reference<typename OutputPortalType::ValueType>::type
ExclusiveScanPortals(InputPortalType inputPortal, OutputPortalType outputPortal)
{
  ExclusiveScanBodyTBB<InputPortalType, OutputPortalType>
      body(inputPortal, outputPortal);
  dax::Id arrayLength = inputPortal.GetNumberOfValues();

  typename boost::remove_reference<typename OutputPortalType::ValueType>::type
      lastValue = inputPortal.Get(arrayLength-1);

  ::tbb::parallel_scan(
        ::tbb::blocked_range<dax::Id>(0, arrayLength, TBB_GRAIN_SIZE),
        body);

  // Seems a little weird to me that we would return the last value in the
  // array rather than the sum, but that is how the function is specified.
  return body.Sum - lastValue;
}

} // namespace detail

template<typename T, class CIn, class COut>
DAX_CONT_EXPORT T InclusiveScan(
    const dax::cont::ArrayHandle<T,CIn,dax::tbb::cont::DeviceAdapterTagTBB>
        &input,
    dax::cont::ArrayHandle<T,COut,dax::tbb::cont::DeviceAdapterTagTBB>
        &output,
    dax::tbb::cont::DeviceAdapterTagTBB)
{
  return detail::InclusiveScanPortals(
        input.PrepareForInput(),
        output.PrepareForOutput(input.GetNumberOfValues()));
}

template<typename T, class CIn, class COut>
DAX_CONT_EXPORT T ExclusiveScan(
    const dax::cont::ArrayHandle<T,CIn,dax::tbb::cont::DeviceAdapterTagTBB>
        &input,
    dax::cont::ArrayHandle<T,COut,dax::tbb::cont::DeviceAdapterTagTBB>
        &output,
    dax::tbb::cont::DeviceAdapterTagTBB)
{
  return detail::ExclusiveScanPortals(
        input.PrepareForInput(),
        output.PrepareForOutput(input.GetNumberOfValues()));
}

template<typename T, class CIn, class CVal, class COut>
DAX_CONT_EXPORT void LowerBounds(
    const dax::cont::ArrayHandle<T,CIn,dax::tbb::cont::DeviceAdapterTagTBB>
        &daxNotUsed(input),
    const dax::cont::ArrayHandle<T,CVal,dax::tbb::cont::DeviceAdapterTagTBB>
        &daxNotUsed(values),
    dax::cont::ArrayHandle<dax::Id,COut,dax::tbb::cont::DeviceAdapterTagTBB>
        &daxNotUsed(output),
    dax::tbb::cont::DeviceAdapterTagTBB)
{
  //TODO
}

template<class CIn, class COut>
DAX_CONT_EXPORT void LowerBounds(
    const dax::cont::ArrayHandle<dax::Id,CIn,dax::tbb::cont::DeviceAdapterTagTBB>
        &input,
    dax::cont::ArrayHandle<dax::Id,COut,dax::tbb::cont::DeviceAdapterTagTBB>
        &values_output,
    dax::tbb::cont::DeviceAdapterTagTBB)
{
  dax::cont::internal::LowerBounds(input,
                                   values_output,
                                   values_output,
                                   dax::tbb::cont::DeviceAdapterTagTBB());
}

namespace detail {

template<class FunctorType>
class ScheduleKernelTBB
{
public:
  DAX_CONT_EXPORT ScheduleKernelTBB(const FunctorType &functor)
    : Functor(functor)
  {  }

  DAX_CONT_EXPORT void SetErrorMessageBuffer(
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->ErrorMessage = errorMessage;
    this->Functor.SetErrorMessageBuffer(errorMessage);
  }

  DAX_EXEC_EXPORT
  void operator()(const ::tbb::blocked_range<dax::Id> &range) const {
    // The TBB device adapter causes array classes to be shared between
    // control and execution environment. This means that it is possible for an
    // exception to be thrown even though this is typically not allowed.
    // Throwing an exception from here is bad because there are several
    // simultaneous threads running. Get around the problem by catching the
    // error and setting the message buffer as expected.
    try
      {
      for (dax::Id index = range.begin(); index < range.end(); index++)
        {
        this->Functor(index);
        }
      }
    catch (dax::cont::Error error)
      {
      this->ErrorMessage.RaiseError(error.GetMessage().c_str());
      }
    catch (...)
      {
      this->ErrorMessage.RaiseError(
          "Unexpected error in execution environment.");
      }
  }

private:
  FunctorType Functor;
  dax::exec::internal::ErrorMessageBuffer ErrorMessage;
};

} // namespace detail

template<class FunctorType>
DAX_CONT_EXPORT void Schedule(FunctorType functor,
                              dax::Id numInstances,
                              dax::tbb::cont::DeviceAdapterTagTBB)
{
  const dax::Id MESSAGE_SIZE = 1024;
  char errorString[MESSAGE_SIZE];
  errorString[0] = '\0';
  dax::exec::internal::ErrorMessageBuffer
      errorMessage(errorString, MESSAGE_SIZE);

  functor.SetErrorMessageBuffer(errorMessage);

  detail::ScheduleKernelTBB<FunctorType> kernel(functor);

  ::tbb::blocked_range<dax::Id> range(0, numInstances, detail::TBB_GRAIN_SIZE);

  ::tbb::parallel_for(range, kernel);

  if (errorMessage.IsErrorRaised())
    {
    throw dax::cont::ErrorExecution(errorString);
    }
}


template<typename T, class Container>
DAX_CONT_EXPORT void Sort(
    dax::cont::ArrayHandle<T,Container,dax::tbb::cont::DeviceAdapterTagTBB>
        &values,
    dax::tbb::cont::DeviceAdapterTagTBB)
{
  typedef typename dax::cont::ArrayHandle<
      T,Container,dax::tbb::cont::DeviceAdapterTagTBB>::PortalExecution
      PortalType;

  PortalType arrayPortal = values.PrepareForInPlace();
  ::tbb::parallel_sort(arrayPortal.GetIteratorBegin(),
                       arrayPortal.GetIteratorEnd());
}

template<typename T, class CStencil, class COut>
DAX_CONT_EXPORT void StreamCompact(
    const dax::cont::ArrayHandle<T,CStencil,dax::tbb::cont::DeviceAdapterTagTBB>
        &daxNotUsed(stencil),
    dax::cont::ArrayHandle<dax::Id,COut,dax::tbb::cont::DeviceAdapterTagTBB>
        &daxNotUsed(output),
    dax::tbb::cont::DeviceAdapterTagTBB)
{
  //TODO
}

template<typename T, typename U, class CIn, class CStencil, class COut>
static void StreamCompact(
    const dax::cont::ArrayHandle<T,CIn,dax::tbb::cont::DeviceAdapterTagTBB>
        &input,
    const dax::cont::ArrayHandle<U,CStencil,dax::tbb::cont::DeviceAdapterTagTBB>
        &stencil,
    dax::cont::ArrayHandle<T,COut,dax::tbb::cont::DeviceAdapterTagTBB> &output,
    dax::tbb::cont::DeviceAdapterTagTBB)
{
  dax::cont::internal::StreamCompactGeneral(
        input, stencil, output, dax::tbb::cont::DeviceAdapterTagTBB());
}

template<typename T, class Container>
DAX_CONT_EXPORT void Unique(
    dax::cont::ArrayHandle<T,Container,dax::tbb::cont::DeviceAdapterTagTBB>
        &daxNotUsed(values),
    dax::tbb::cont::DeviceAdapterTagTBB)
{
  //TODO
}


}
}
} // namespace dax::cont::internal

#endif //__dax_tbb_cont_internal_DeviceAdapterAlgorithmTBB_h
