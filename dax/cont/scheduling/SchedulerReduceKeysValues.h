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
#ifndef __dax_cont_scheduling_SchedulerReduceKeysValues_h
#define __dax_cont_scheduling_SchedulerReduceKeysValues_h

#include <dax/Types.h>
#include <dax/CellTraits.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/scheduling/AddReduceKeysArgs.h>
#include <dax/cont/scheduling/SchedulerDefault.h>
#include <dax/cont/scheduling/SchedulerTags.h>
#include <dax/cont/scheduling/VerifyUserArgLength.h>
#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/Tag.h>
#include <dax/cont/sig/VisitIndex.h>
#include <dax/math/Compare.h>

#include <dax/exec/internal/kernel/GenerateWorklets.h>

#if !(__cplusplus >= 201103L)
# include <dax/internal/ParameterPackCxx03.h>
#endif // !(__cplusplus >= 201103L)

namespace dax { namespace cont { namespace scheduling {


template <class DeviceAdapterTag>
class Scheduler<DeviceAdapterTag,dax::cont::scheduling::ReduceKeysValuesTag>
{

public:
  //default constructor so we can insantiate const schedulers
  DAX_CONT_EXPORT Scheduler():DefaultScheduler(){}

  //copy constructor so that people can pass schedulers around by value
  DAX_CONT_EXPORT Scheduler(
      const Scheduler<DeviceAdapterTag,
          dax::cont::scheduling::ReduceKeysValuesTag>& other ):
  DefaultScheduler(other.DefaultScheduler)
  {
  }

  template <class WorkletType, typename ParameterPackType>
  DAX_CONT_EXPORT void Invoke(WorkletType worklet,
                              ParameterPackType& args) const
  {
    typedef typename WorkletType::WorkletType RealWorkletType;
    typedef dax::cont::scheduling::VerifyUserArgLength<RealWorkletType,
                ParameterPackType::NUM_PARAMETERS> WorkletUserArgs;

    //if you are getting this error you are passing less arguments than requested
    //in the control signature of this worklet
    DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::NotEnoughParameters));

    //if you are getting this error you are passing too many arguments
    //than requested in the control signature of this worklet
    DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::TooManyParameters));

    this->DoReduce(worklet, args);
  }

private:
  typedef dax::cont::scheduling::Scheduler<DeviceAdapterTag,
    dax::cont::scheduling::ScheduleDefaultTag> SchedulerDefaultType;
  const SchedulerDefaultType DefaultScheduler;

  template <class WorkletType,
            typename KeysHandleType,
            typename ParameterPackType>
  DAX_CONT_EXPORT void DoReduce(
      dax::cont::ReduceKeysValues<
        WorkletType, KeysHandleType>& workletWrapper,
      const ParameterPackType &arguments) const
  {
    typedef dax::cont::ReduceKeysValues<WorkletType,KeysHandleType>
      WorkletWrapperType;

    typedef dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag> Algorithm;

    typedef typename WorkletWrapperType::ReductionMapType ReductionMapType;
    typedef typename WorkletWrapperType::KeysType KeysType;

    //Get a map from output indices to input groups.
    workletWrapper.BuildReductionMap();
    if (workletWrapper.GetReleaseKeys())
      {
      workletWrapper.DoReleaseKeys();
      }

    ReductionMapType reductionCounts = workletWrapper.GetReductionCounts();
    ReductionMapType reductionOffsets = workletWrapper.GetReductionOffsets();
    ReductionMapType reductionIndices = workletWrapper.GetReductionIndices();
    KeysType reductionKeys = workletWrapper.GetReductionKeys();

    // We need to scan the args of the reduce keys/values worklet and determine
    // if we have the ReductionCount/Offset/Index signature.  The control signature needs to
    // be modified to add this array to the arguments and the execution
    // signature has to be modified to ensure that the ReductionCount signature
    // points to the appropriate array.  The AddReduceKeysArgs does all this.
    typedef typename dax::cont::scheduling::AddReduceKeysArgs<
                  WorkletType>::DerivedWorkletType DerivedWorkletType;

    //we get our magic here. we need to wrap some parameters and pass
    //them to the real scheduler
    DerivedWorkletType derivedWorklet(workletWrapper.GetWorklet());
    this->DefaultScheduler.Invoke(derivedWorklet,
                                  arguments.Append(reductionCounts)
                                  .Append(reductionOffsets)
                                  .Append(reductionIndices)
                                  );
  }
};

} } }

#endif //__dax_cont_scheduling_ReduceKeysValues_h
