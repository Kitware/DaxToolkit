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

#include <tbb/blocked_range2d.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <tbb/parallel_sort.h>

namespace dax {
namespace cont {
namespace internal {

template<>
struct DeviceAdapterAlgorithm<dax::tbb::cont::DeviceAdapterTagTBB> :
    DeviceAdapterAlgorithmGeneral<
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
    ScanInclusiveBody(const ScanInclusiveBody &body, ::tbb::split)
      : Sum(0),
        InputPortal(body.InputPortal),
        OutputPortal(body.OutputPortal) {  }

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
    ::tbb::parallel_scan(
          ::tbb::blocked_range<dax::Id>(0, arrayLength, TBB_GRAIN_SIZE),
          body);
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
    ScanExclusiveBody(const ScanExclusiveBody &body, ::tbb::split)
      : Sum(0),
        InputPortal(body.InputPortal),
        OutputPortal(body.OutputPortal) {  }

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

    ::tbb::parallel_scan(
          ::tbb::blocked_range<dax::Id>(0, arrayLength, TBB_GRAIN_SIZE),
          body);

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

  template<class FunctorType, class GridType>
  DAX_CONT_EXPORT
  static void ScheduleOnCells(FunctorType functor, dax::Id count, GridType)
  {
    DeviceAdapterAlgorithm<dax::tbb::cont::DeviceAdapterTagTBB>::Schedule(
                                                                  functor,
                                                                  count);
  }

  template<class FunctorType>
  class ScheduleUniformGridKernel
  {
  public:
    DAX_CONT_EXPORT ScheduleUniformGridKernel(const FunctorType &functor,
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
    void operator()(const ::tbb::blocked_range2d<dax::Id> &range) const {
      try
      {
      dax::exec::internal::IJKIndex index(this->Dims);
      for( dax::Id k=range.rows().begin(); k!=range.rows().end(); ++k)
        {
        index.setK(k);
        for( dax::Id j=range.cols().begin(); j!=range.cols().end(); ++j)
          {
          index.setJ(j);
          for (dax::Id i=0; i < this->Dims[0]; ++i)
            {
            index.setI(i);
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

  template<class FunctorType, class GridType>
  DAX_CONT_EXPORT
  static void ScheduleOnCells(FunctorType functor,
                       dax::Id3 dims,
                       GridType)
  {
    //we need to extract from the functor that uniform grid information
    const dax::Id MESSAGE_SIZE = 1024;
    char errorString[MESSAGE_SIZE];
    errorString[0] = '\0';
    dax::exec::internal::ErrorMessageBuffer
        errorMessage(errorString, MESSAGE_SIZE);

    functor.SetErrorMessageBuffer(errorMessage);

    //4: use the extents and number of cells to determine the blocked_rang2
    //that will best suited for this grid. Presumption is that we are 3d
    //in the future we will have to make special calls for 2d uniform grids
    ::tbb::blocked_range2d<dax::Id> range(0,
                                          dims[2],
                                          0,
                                          dims[1]);

    ScheduleUniformGridKernel<FunctorType> kernel(functor,dims);
    ::tbb::parallel_for(range, kernel);

    if (errorMessage.IsErrorRaised())
      {
      throw dax::cont::ErrorExecution(errorString);
      }
  }

  template<class FunctorType>
  DAX_CONT_EXPORT
  static void Schedule(FunctorType functor, dax::Id,
                       dax::cont::internal::UniformGridTag)
  {
    //we need to extract from the functor that uniform grid information
    const dax::Id MESSAGE_SIZE = 1024;
    char errorString[MESSAGE_SIZE];
    errorString[0] = '\0';
    dax::exec::internal::ErrorMessageBuffer
        errorMessage(errorString, MESSAGE_SIZE);

    functor.SetErrorMessageBuffer(errorMessage);


    //1: find the topology argument
    typedef typename FunctorType::BindingsType BindingsType;
    typedef typename dax::cont::internal::FindBinding<
                                    BindingsType,
                                    dax::cont::arg::Topology>::type TopoIndex;
    typedef typename BindingsType::template GetType<
                            TopoIndex::value>::type TopoControlBinding;

    //2: now that we have the execution arg object we can extract the grid
    typedef typename TopoControlBinding::ExecArg TopoExecArgType;
    TopoExecArgType TopoExecArg;

    //3: have the grid, extract the extents
    const dax::Extent3& extents = TopoExecArg.Extent;

    //4: use the extents and number of cells to determine the blocked_rang2
    //that will best suited for this grid. Presumption is that we are 3d
    //in the future we will have to make special calls for 2d uniform grids
    ::tbb::blocked_range2d<dax::Id> range(extents.Min[0],
                                          extents.Max[0],
                                          extents.Min[1],
                                          extents.Max[1]);

    ScheduleUniformGridKernel<FunctorType> kernel(functor,extents);
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

};

}
}
} // namespace dax::cont::internal

#endif //__dax_tbb_cont_internal_DeviceAdapterAlgorithmTBB_h
