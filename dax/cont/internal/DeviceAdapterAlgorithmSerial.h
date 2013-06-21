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

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ErrorExecution.h>
#include <dax/cont/internal/DeviceAdapterAlgorithm.h>
#include <dax/cont/internal/DeviceAdapterAlgorithmGeneral.h>
#include <dax/cont/internal/DeviceAdapterTagSerial.h>

#include <dax/exec/internal/IJKIndex.h>
#include <dax/exec/internal/ErrorMessageBuffer.h>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/utility/enable_if.hpp>

#include <algorithm>
#include <numeric>

namespace dax {
namespace cont {
namespace internal {


template<>
struct DeviceAdapterAlgorithm<dax::cont::DeviceAdapterTagSerial> :
    DeviceAdapterAlgorithmGeneral<
        DeviceAdapterAlgorithm<dax::cont::DeviceAdapterTagSerial>,
        dax::cont::DeviceAdapterTagSerial>
{

public:
  template<typename T, class CIn, class COut>
  DAX_CONT_EXPORT static T ScanInclusive(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTagSerial> &input,
      dax::cont::ArrayHandle<T,COut,DeviceAdapterTagSerial>& output)
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

  template<typename T, class CIn, class COut>
  DAX_CONT_EXPORT static T ScanExclusive(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTagSerial> &input,
      dax::cont::ArrayHandle<T,COut,DeviceAdapterTagSerial>& output)
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

    T fullSum = outputPortal.Get(numberOfValues - 1);

    // Shift right by one
    std::copy_backward(outputPortal.GetIteratorBegin(),
                       outputPortal.GetIteratorEnd()-1,
                       outputPortal.GetIteratorEnd());
    outputPortal.Set(0, 0);
    return fullSum;
  }

private:
  // This runs in the execution environment.
  template<class FunctorType>
  class ScheduleKernel
  {
  public:
    ScheduleKernel(const FunctorType &functor)
      : Functor(functor) {  }

    //needed for when calling from schedule on a range
    DAX_EXEC_EXPORT void operator()(dax::Id index) const
    {
      this->Functor(index);
    }

  private:
    const FunctorType Functor;
  };

public:
  template<class Functor>
  DAX_CONT_EXPORT static void Schedule(Functor functor,
                                       dax::Id numInstances)
  {
    const dax::Id MESSAGE_SIZE = 1024;
    char errorString[MESSAGE_SIZE];
    errorString[0] = '\0';
    dax::exec::internal::ErrorMessageBuffer
        errorMessage(errorString, MESSAGE_SIZE);

    functor.SetErrorMessageBuffer(errorMessage);

    DeviceAdapterAlgorithm::ScheduleKernel<Functor> kernel(functor);

    std::for_each(
          ::boost::counting_iterator<dax::Id>(0),
          ::boost::counting_iterator<dax::Id>(numInstances),
          kernel);

    if (errorMessage.IsErrorRaised())
      {
      throw dax::cont::ErrorExecution(errorString);
      }
  }

  template<class FunctorType>
  DAX_CONT_EXPORT
  static void Schedule(FunctorType functor, dax::Id3 rangeMax)
  {
    const dax::Id MESSAGE_SIZE = 1024;
    char errorString[MESSAGE_SIZE];
    errorString[0] = '\0';
    dax::exec::internal::ErrorMessageBuffer
        errorMessage(errorString, MESSAGE_SIZE);

    functor.SetErrorMessageBuffer(errorMessage);

    dax::exec::internal::IJKIndex index(rangeMax);
    for( dax::Id k=0; k!=rangeMax[2]; ++k)
      {
      index.SetK(k);
      for( dax::Id j=0; j!=rangeMax[1]; ++j)
        {
        index.SetJ(j);
        for (dax::Id i=0; i < rangeMax[0]; ++i)
          {
          index.SetI(i);
          functor(index);
          }
        }
      }
    if (errorMessage.IsErrorRaised())
      {
      throw dax::cont::ErrorExecution(errorString);
      }
  }

  template<typename T, class Container>
  DAX_CONT_EXPORT static void Sort(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>& values)
  {
    typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>
        ::PortalExecution PortalType;

    PortalType arrayPortal = values.PrepareForInPlace();
    std::sort(arrayPortal.GetIteratorBegin(), arrayPortal.GetIteratorEnd());
  }

  template<typename T, class Container, class Compare>
  DAX_CONT_EXPORT static void Sort(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>& values,
      Compare comp)
  {
    typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial>
        ::PortalExecution PortalType;

    PortalType arrayPortal = values.PrepareForInPlace();
    std::sort(arrayPortal.GetIteratorBegin(), arrayPortal.GetIteratorEnd(),comp);
  }

  DAX_CONT_EXPORT static void Synchronize()
  {
    // Nothing to do. This device is serial and has no asynchronous operations.
  }

};

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_DeviceAdapterAlgorithmSerial_h
