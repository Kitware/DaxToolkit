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
//  Copyright 2013 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//===========================================x==================================
#ifndef __dax_cont_scheduling_SchedulerGenerateKeysValues_h
#define __dax_cont_scheduling_SchedulerGenerateKeysValues_h

#include <dax/Types.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/Tag.h>
#include <dax/cont/sig/VisitIndex.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/ArrayHandleCounting.h>
#include <dax/cont/scheduling/SchedulerTags.h>
#include <dax/cont/scheduling/SchedulerDefault.h>
#include <dax/cont/scheduling/VerifyUserArgLength.h>
#include <dax/cont/scheduling/AddVisitIndexArg.h>

#include <dax/internal/ParameterPack.h>

#include <dax/exec/internal/kernel/GenerateWorklets.h>

#if !(__cplusplus >= 201103L)
# include <dax/internal/ParameterPackCxx03.h>
#endif // !(__cplusplus >= 201103L)

namespace dax { namespace cont { namespace scheduling {

template <class DeviceAdapterTag>
class Scheduler<DeviceAdapterTag,dax::cont::scheduling::GenerateKeysValuesTag>
{
public:
  //default constructor so we can insantiate const schedulers
  DAX_CONT_EXPORT Scheduler():DefaultScheduler(){}

  //copy constructor so that people can pass schedulers around by value
  DAX_CONT_EXPORT Scheduler(
      const Scheduler<DeviceAdapterTag,
          dax::cont::scheduling::GenerateKeysValuesTag>& other ):
  DefaultScheduler(other.DefaultScheduler)
  {
  }

  template <class WorkletType, typename ParameterPackType>
  DAX_CONT_EXPORT void Invoke(WorkletType w,
                              ParameterPackType &args) const
  {
    //we are being passed dax::cont::GenerateKeysValues,
    //we want the actual exec worklet that is being passed to scheduleGenerateTopo
    typedef typename WorkletType::WorkletType RealWorkletType;
    typedef dax::cont::scheduling::VerifyUserArgLength<RealWorkletType,
                ParameterPackType::NUM_PARAMETERS> WorkletUserArgs;

  //if you are getting this error you are passing less arguments than requested
  //in the control signature of this worklet
  DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::NotEnoughParameters));

  //if you are getting this error you are passing too many arguments
  //than requested in the control signature of this worklet
  DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::TooManyParameters));

  this->InvokeGenerateKeysValues(w,
                          dax::internal::ParameterPackGetArgument<1>(args),
                          args);
  }

private:
  typedef dax::cont::scheduling::Scheduler<DeviceAdapterTag,
    dax::cont::scheduling::ScheduleDefaultTag> SchedulerDefaultType;
  const SchedulerDefaultType DefaultScheduler;

//want the basic implementation to be easily edited, instead of inside
//the BOOST_PP block and unreadable. This version of InvokeGenerateKeysValues
//handles the use case no parameters
template <class WorkletType,
          typename ClassifyHandleType,
          typename InputGrid,
          typename ParameterPackType>
DAX_CONT_EXPORT void InvokeGenerateKeysValues(
    dax::cont::GenerateKeysValues<
    WorkletType,ClassifyHandleType>& workletWrapper,
    const InputGrid inputGrid,
    const ParameterPackType &arguments) const
  {
  typedef dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag> Algorithm;
  typedef dax::cont::ArrayHandle<dax::Id, ArrayContainerControlTagBasic,
      DeviceAdapterTag> IdArrayHandleType;

  //do an inclusive scan of the cell count / cell mask to get the number
  //of cells in the output
  IdArrayHandleType scannedOutputCounts;
  const dax::Id numNewValues =
      Algorithm::ScanInclusive(workletWrapper.GetOutputCountArray(),
                               scannedOutputCounts);

  if(workletWrapper.GetReleaseOutputCountArray())
    {
    workletWrapper.DoReleaseOutputCountArray();
    }

  if(numNewValues == 0)
    {
    //nothing to do
    return;
    }


  //now do the lower bounds of the cell indices so that we figure out
  IdArrayHandleType outputIndexRanges;
  Algorithm::UpperBounds(scannedOutputCounts,
                 dax::cont::make_ArrayHandleCounting(dax::Id(0),numNewValues),
                 outputIndexRanges);

  // We are done with scannedOutputCounts.
  scannedOutputCounts.ReleaseResources();

  //we need to scan the args of the generate topology worklet and determine if
  //we have the VisitIndex signature. If we do, we have to call a different
  //Invoke algorithm, which properly uploads the visitIndex information. Since
  //this information is slow to compute we don't want to always upload the
  //information, instead only compute when explicitly requested

  //The AddVisitIndexArg does all this, plus creates a derived worklet
  //from the users worklet with the visit index added to the signature.
  typedef dax::cont::scheduling::AddVisitIndexArg<WorkletType,
    Algorithm,IdArrayHandleType> AddVisitIndexFunctor;
  typedef typename AddVisitIndexFunctor::VisitIndexArgType IndexArgType;
  typedef typename AddVisitIndexFunctor::DerivedWorkletType DerivedWorkletType;

  IndexArgType visitIndex;
  AddVisitIndexFunctor createVisitIndex;
  createVisitIndex(this->DefaultScheduler,outputIndexRanges,visitIndex);

  DerivedWorkletType derivedWorklet(workletWrapper.GetWorklet());

  //we get our magic here. we need to wrap some paramemters and pass
  //them to the real scheduler
  this->DefaultScheduler.Invoke(
        derivedWorklet,
        arguments.template Replace<1>(
            dax::cont::make_Permutation(outputIndexRanges,inputGrid,
                                        inputGrid.GetNumberOfCells()))
        .Append(visitIndex));
  }

};

} } }
#endif //__dax_cont_scheduling_SchedulerGenerateKeysValues_h
