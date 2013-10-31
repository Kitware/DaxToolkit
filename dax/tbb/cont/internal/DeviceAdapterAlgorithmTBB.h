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


#include <dax/cont/internal/IteratorFromArrayPortal.h>


#include <dax/tbb/cont/internal/DeviceAdapterTagTBB.h>
#include <dax/tbb/cont/internal/ArrayManagerExecutionTBB.h>

#include <dax/exec/internal/ErrorMessageBuffer.h>

#include <dax/Extent.h>
#include <dax/cont/arg/Topology.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ErrorExecution.h>
#include <dax/cont/internal/DeviceAdapterAlgorithm.h>
#include <dax/cont/internal/DeviceAdapterAlgorithmGeneral.h>
#include <dax/cont/internal/FindBinding.h>
#include <dax/cont/internal/GridTags.h>

#include <dax/exec/internal/IJKIndex.h>
#include <boost/type_traits/remove_reference.hpp>


//we provide an patched implementation of tbb parallel_sort
//that fixes ADL for std::swap. This patch has been submitted to Intel
//and should be included in future version of TBB.
#include <dax/tbb/cont/internal/parallel_sort.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range3d.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <tbb/partitioner.h>
#include <tbb/tick_count.h>


namespace dax {
namespace cont {

template<>
struct DeviceAdapterAlgorithm<dax::tbb::cont::DeviceAdapterTagTBB> :
    dax::cont::internal::DeviceAdapterAlgorithmGeneral<
        DeviceAdapterAlgorithm<dax::tbb::cont::DeviceAdapterTagTBB>,
        dax::tbb::cont::DeviceAdapterTagTBB>
{
private:
  // The "grain size" of scheduling with TBB.  Not a lot of thought has gone
  // into picking this size.
  static const dax::Id TBB_GRAIN_SIZE = 128;

  template<class InputPortalType, class OutputPortalType>
  struct ScanInclusiveBody
  {
    typedef typename boost::remove_reference<
        typename OutputPortalType::ValueType>::type ValueType;
    ValueType Sum;
    InputPortalType InputPortal;
    OutputPortalType OutputPortal;

    DAX_CONT_EXPORT
    ScanInclusiveBody(const InputPortalType &inputPortal,
                      const OutputPortalType &outputPortal)
      : Sum(ValueType(0)), InputPortal(inputPortal), OutputPortal(outputPortal)
    {  }

    DAX_EXEC_CONT_EXPORT
    ScanInclusiveBody(const ScanInclusiveBody &body, ::tbb::split)
      : Sum(0),
        InputPortal(body.InputPortal),
        OutputPortal(body.OutputPortal) {  }

    DAX_EXEC_EXPORT
    void operator()(const ::tbb::blocked_range<dax::Id> &range, ::tbb::pre_scan_tag)
    {
      typedef typename InputPortalType::IteratorType InIterator;

      //use temp, and iterators instead of member variable to reduce false sharing
      ValueType temp = this->Sum;
      InIterator inIter = this->InputPortal.GetIteratorBegin() + range.begin();
      for (dax::Id index = range.begin(); index != range.end();
           ++index, ++inIter)
        {
        temp = temp + *inIter;
        }
      this->Sum = temp;
    }

    DAX_EXEC_EXPORT
    void operator()(const ::tbb::blocked_range<dax::Id> &range, ::tbb::final_scan_tag)
    {
      typedef typename InputPortalType::IteratorType InIterator;
      typedef typename OutputPortalType::IteratorType OutIterator;

      //use temp, and iterators instead of member variable to reduce false sharing
      ValueType temp = this->Sum;
      InIterator inIter = this->InputPortal.GetIteratorBegin() + range.begin();
      OutIterator outIter = this->OutputPortal.GetIteratorBegin() + range.begin();
      for (dax::Id index = range.begin(); index != range.end();
           ++index, ++inIter, ++outIter)
        {
        *outIter = temp = (temp + *inIter);
        }
      this->Sum = temp;
    }

    DAX_EXEC_CONT_EXPORT
    void reverse_join(const ScanInclusiveBody &left)
    {
      this->Sum = left.Sum + this->Sum;
    }

    DAX_EXEC_CONT_EXPORT
    void assign(const ScanInclusiveBody &src)
    {
      this->Sum = src.Sum;
    }
  };

  template<class InputPortalType, class OutputPortalType>
  DAX_CONT_EXPORT static
  typename boost::remove_reference<typename OutputPortalType::ValueType>::type
  ScanInclusivePortals(InputPortalType inputPortal,
                       OutputPortalType outputPortal)
  {
    ScanInclusiveBody<InputPortalType, OutputPortalType>
        body(inputPortal, outputPortal);
    dax::Id arrayLength = inputPortal.GetNumberOfValues();
    ::tbb::parallel_scan( ::tbb::blocked_range<dax::Id>(0, arrayLength), body);
    return body.Sum;
  }

  template<class InputPortalType, class OutputPortalType>
  struct ScanExclusiveBody
  {
    typedef typename boost::remove_reference<
        typename OutputPortalType::ValueType>::type ValueType;
    ValueType Sum;
    InputPortalType InputPortal;
    OutputPortalType OutputPortal;

    DAX_CONT_EXPORT
    ScanExclusiveBody(const InputPortalType &inputPortal,
                      const OutputPortalType &outputPortal)
      : Sum(ValueType(0)), InputPortal(inputPortal), OutputPortal(outputPortal)
    {  }

    DAX_EXEC_CONT_EXPORT
    ScanExclusiveBody(const ScanExclusiveBody &body, ::tbb::split)
      : Sum(0),
        InputPortal(body.InputPortal),
        OutputPortal(body.OutputPortal) {  }

    DAX_EXEC_EXPORT
    void operator()(const ::tbb::blocked_range<dax::Id> &range, ::tbb::pre_scan_tag)
    {
      typedef typename InputPortalType::IteratorType InIterator;

      ValueType temp = this->Sum;

      //move the iterator to the first item
      InIterator iter = this->InputPortal.GetIteratorBegin() + range.begin();
      for (dax::Id index = range.begin(); index != range.end(); ++index, ++iter)
        {
        temp = temp + *iter;
        }
      this->Sum = temp;
    }

    DAX_EXEC_EXPORT
    void operator()(const ::tbb::blocked_range<dax::Id> &range, ::tbb::final_scan_tag)
    {
      typedef typename InputPortalType::IteratorType InIterator;
      typedef typename OutputPortalType::IteratorType OutIterator;

      ValueType temp = this->Sum;

      //move the iterators to the first item
      InIterator inIter = this->InputPortal.GetIteratorBegin() + range.begin();
      OutIterator outIter = this->OutputPortal.GetIteratorBegin() + range.begin();
      for (dax::Id index = range.begin(); index != range.end();
           ++index, ++inIter, ++outIter)
        {
        //copy into a local reference since Input and Output portal
        //could point to the same memory location
        ValueType v = *inIter;
        *outIter = temp;
        temp = temp + v;
        }
      this->Sum = temp;
    }

    DAX_EXEC_CONT_EXPORT
    void reverse_join(const ScanExclusiveBody &left)
    {
      this->Sum = left.Sum + this->Sum;
    }

    DAX_EXEC_CONT_EXPORT
    void assign(const ScanExclusiveBody &src)
    {
      this->Sum = src.Sum;
    }
  };

  template<class InputPortalType, class OutputPortalType>
  DAX_CONT_EXPORT static
  typename boost::remove_reference<typename OutputPortalType::ValueType>::type
  ScanExclusivePortals(InputPortalType inputPortal,
                       OutputPortalType outputPortal)
  {
    ScanExclusiveBody<InputPortalType, OutputPortalType>
        body(inputPortal, outputPortal);
    dax::Id arrayLength = inputPortal.GetNumberOfValues();

    ::tbb::parallel_scan( ::tbb::blocked_range<dax::Id>(0, arrayLength), body);

    // Seems a little weird to me that we would return the last value in the
    // array rather than the sum, but that is how the function is specified.
    return body.Sum;
  }

public:
  template<typename T, class CIn, class COut>
  DAX_CONT_EXPORT static T ScanInclusive(
      const dax::cont::ArrayHandle<T,CIn,dax::tbb::cont::DeviceAdapterTagTBB>
          &input,
      dax::cont::ArrayHandle<T,COut,dax::tbb::cont::DeviceAdapterTagTBB>
          &output)
  {
    return ScanInclusivePortals(
          input.PrepareForInput(),
          output.PrepareForOutput(input.GetNumberOfValues()));
  }

  template<typename T, class CIn, class COut>
  DAX_CONT_EXPORT static T ScanExclusive(
      const dax::cont::ArrayHandle<T,CIn,dax::tbb::cont::DeviceAdapterTagTBB>
          &input,
      dax::cont::ArrayHandle<T,COut,dax::tbb::cont::DeviceAdapterTagTBB>
          &output)
  {
    return ScanExclusivePortals(
          input.PrepareForInput(),
          output.PrepareForOutput(input.GetNumberOfValues()));
  }

private:
  template<class FunctorType>
  class ScheduleKernel
  {
  public:
    DAX_CONT_EXPORT ScheduleKernel(const FunctorType &functor)
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

public:
  template<class FunctorType>
  DAX_CONT_EXPORT
  static void Schedule(FunctorType functor, dax::Id numInstances)
  {
    const dax::Id MESSAGE_SIZE = 1024;
    char errorString[MESSAGE_SIZE];
    errorString[0] = '\0';
    dax::exec::internal::ErrorMessageBuffer
        errorMessage(errorString, MESSAGE_SIZE);

    functor.SetErrorMessageBuffer(errorMessage);

    ScheduleKernel<FunctorType> kernel(functor);

    ::tbb::blocked_range<dax::Id> range(0, numInstances, TBB_GRAIN_SIZE);

    ::tbb::parallel_for(range, kernel);

    if (errorMessage.IsErrorRaised())
      {
      throw dax::cont::ErrorExecution(errorString);
      }
  }

private:
  template<class FunctorType>
  class ScheduleKernelId3
  {
  public:
    DAX_CONT_EXPORT ScheduleKernelId3(const FunctorType &functor,
                                      const dax::Id3& dims)
      : Functor(functor),
        Dims(dims)
      {  }

    DAX_CONT_EXPORT void SetErrorMessageBuffer(
        const dax::exec::internal::ErrorMessageBuffer &errorMessage)
    {
      this->ErrorMessage = errorMessage;
      this->Functor.SetErrorMessageBuffer(errorMessage);
    }

    DAX_EXEC_EXPORT
    void operator()(const ::tbb::blocked_range3d<dax::Id> &range) const {
      try
      {
      dax::exec::internal::IJKIndex index(this->Dims);
      for( dax::Id k=range.pages().begin(); k!=range.pages().end(); ++k)
        {
        index.SetK(k);
        for( dax::Id j=range.rows().begin(); j!=range.rows().end(); ++j)
          {
          index.SetJ(j);
          for( dax::Id i=range.cols().begin(); i!=range.cols().end(); ++i)
            {
            index.SetI(i);
            this->Functor(index);
            }
          }
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
    dax::Id3 Dims;
    dax::exec::internal::ErrorMessageBuffer ErrorMessage;
  };

public:
  template<class FunctorType>
  DAX_CONT_EXPORT
  static void Schedule(FunctorType functor,
                       dax::Id3 rangeMax)
  {
    //we need to extract from the functor that uniform grid information
    const dax::Id MESSAGE_SIZE = 1024;
    char errorString[MESSAGE_SIZE];
    errorString[0] = '\0';
    dax::exec::internal::ErrorMessageBuffer
        errorMessage(errorString, MESSAGE_SIZE);

    functor.SetErrorMessageBuffer(errorMessage);

    //memory is generally setup in a way that iterating the first range
    //in the tightest loop has the best cache coherence.
    ::tbb::blocked_range3d<dax::Id> range(0, rangeMax[2],
                                          0, rangeMax[1],
                                          0, rangeMax[0]);

    ScheduleKernelId3<FunctorType> kernel(functor,rangeMax);
    ::tbb::parallel_for(range, kernel);

    if (errorMessage.IsErrorRaised())
      {
      throw dax::cont::ErrorExecution(errorString);
      }
  }

  template<typename T, class Container>
  DAX_CONT_EXPORT static void Sort(
      dax::cont::ArrayHandle<T,Container,dax::tbb::cont::DeviceAdapterTagTBB>
          &values)
  {
    typedef typename dax::cont::ArrayHandle<
        T,Container,dax::tbb::cont::DeviceAdapterTagTBB>::PortalExecution
        PortalType;

    PortalType arrayPortal = values.PrepareForInPlace();
    ::tbb::parallel_sort(arrayPortal.GetIteratorBegin(),
                         arrayPortal.GetIteratorEnd());
  }

  template<typename T, class Container, class Compare>
  DAX_CONT_EXPORT static void Sort(
      dax::cont::ArrayHandle<T,Container,dax::tbb::cont::DeviceAdapterTagTBB>
          &values,
      Compare comp)
  {
    typedef typename dax::cont::ArrayHandle<
        T,Container,dax::tbb::cont::DeviceAdapterTagTBB>::PortalExecution
        PortalType;

    PortalType arrayPortal = values.PrepareForInPlace();
    ::tbb::parallel_sort(arrayPortal.GetIteratorBegin(),
                         arrayPortal.GetIteratorEnd(),
                         comp);
  }


  DAX_CONT_EXPORT static void Synchronize()
  {
    // Nothing to do. This device schedules all of its operations using a
    // split/join paradigm. This means that the if the control threaad is
    // calling this method, then nothing should be running in the execution
    // environment.
  }

};

/// TBB contains its own high resolution timer.
///
template<>
class DeviceAdapterTimerImplementation<dax::tbb::cont::DeviceAdapterTagTBB>
{
public:
  DAX_CONT_EXPORT DeviceAdapterTimerImplementation()
  {
    this->Reset();
  }
  DAX_CONT_EXPORT void Reset()
  {
    dax::cont::DeviceAdapterAlgorithm<
        dax::tbb::cont::DeviceAdapterTagTBB>::Synchronize();
    this->StartTime = ::tbb::tick_count::now();
  }
  DAX_CONT_EXPORT dax::Scalar GetElapsedTime()
  {
    dax::cont::DeviceAdapterAlgorithm<
        dax::tbb::cont::DeviceAdapterTagTBB>::Synchronize();
    ::tbb::tick_count currentTime = ::tbb::tick_count::now();
    ::tbb::tick_count::interval_t elapsedTime = currentTime - this->StartTime;
    return static_cast<dax::Scalar>(elapsedTime.seconds());
  }

private:
  ::tbb::tick_count StartTime;
};

}
} // namespace dax::cont

#endif //__dax_tbb_cont_internal_DeviceAdapterAlgorithmTBB_h
