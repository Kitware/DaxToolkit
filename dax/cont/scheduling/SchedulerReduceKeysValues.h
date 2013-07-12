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
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef __dax_cont_scheduling_SchedulerReduceKeysValues_h
#define __dax_cont_scheduling_SchedulerReduceKeysValues_h

#include <dax/Types.h>
#include <dax/CellTraits.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/scheduling/AddReductionCountArg.h>
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
  typedef dax::cont::scheduling::Scheduler<DeviceAdapterTag,
    dax::cont::scheduling::ScheduleDefaultTag> SchedulerDefaultType;
  const SchedulerDefaultType DefaultScheduler;

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

#if __cplusplus >= 201103L
  template <class WorkletType, _dax_pp_typename___T>
  DAX_CONT_EXPORT void Invoke(WorkletType w, T...a)
  {
    typedef dax::cont::scheduling::VerifyUserArgLength<WorkletType,
              sizeof...(T)> WorkletUserArgs;
    //if you are getting this error you are passing less arguments than requested
    //in the control signature of this worklet
    DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::NotEnoughParameters));

    //if you are getting this error you are passing too many arguments
    //than requested in the control signature of this worklet
    DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::TooManyParameters));

    this->DoReduce(w,T...);
  }

  //todo implement the DoReduce method with C11 syntax

#else
# define BOOST_PP_ITERATION_PARAMS_1 (3, (2, 10, <dax/cont/scheduling/SchedulerReduceKeysValues.h>))
# include BOOST_PP_ITERATE()
#endif
};

} } }

#endif //__dax_cont_scheduling_ReduceKeysValues_h

#else // defined(BOOST_PP_IS_ITERATING)
public: //needed so that each iteration of invoke is public
  template <class WorkletType, _dax_pp_typename___T>
  DAX_CONT_EXPORT void Invoke(WorkletType w, _dax_pp_params___(a)) const
  {
    //we are being passed dax::cont::ReduceKeysValues,
    //we want the actual exec worklet that is being passed to
    //ScheduleReduceKeyValues
    typedef typename WorkletType::WorkletType RealWorkletType;
    typedef dax::cont::scheduling::VerifyUserArgLength<RealWorkletType,
              _dax_pp_sizeof___T> WorkletUserArgs;
    //if you are getting this error you are passing less arguments than requested
    //in the control signature of this worklet
    DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::NotEnoughParameters));

    //if you are getting this error you are passing too many arguments
    //than requested in the control signature of this worklet
    DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::TooManyParameters));

    this->DoReduce(w,_dax_pp_args___(a));
  }

private:
  template <class WorkletType,
            typename KeysHandleType,
            _dax_pp_typename___T>
  DAX_CONT_EXPORT void DoReduce(
      dax::cont::ReduceKeysValues<WorkletType,KeysHandleType>& workletWrapper,
      _dax_pp_params___(a)) const
  {
    typedef dax::cont::ReduceKeysValues<WorkletType,KeysHandleType>
      WorkletWrapperType;

    typedef dax::cont::internal::DeviceAdapterAlgorithm<DeviceAdapterTag>
      Algorithm;

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
    // points to the appropriate array.  The AddReductionCountArg does all this.
    typedef dax::cont::scheduling::AddReductionCountArg<WorkletType>
      AddCounts;
    typedef typename AddCounts::DerivedWorkletType AddCountsWorkletType;
    AddCountsWorkletType workletWithCounts(workletWrapper.GetWorklet());

    typedef dax::cont::scheduling::AddReductionOffsetArg<AddCountsWorkletType>
      AddOffsets;
    typedef typename AddOffsets::DerivedWorkletType AddOffsetsWorkletType;
    AddOffsetsWorkletType workletWithOffsets(workletWithCounts);

    typedef dax::cont::scheduling::AddReductionIndexPortalArg<AddOffsetsWorkletType>
      AddIndices;
    typedef typename AddIndices::DerivedWorkletType AddIndicesWorkletType;
    typedef AddIndicesWorkletType DerivedWorkletType;

    //we get our magic here. we need to wrap some parameters and pass
    //them to the real scheduler
    DerivedWorkletType derivedWorklet(workletWithOffsets);
    this->DefaultScheduler.Invoke(derivedWorklet,
                                  _dax_pp_args___(a),
                                  reductionCounts,
                                  reductionOffsets,
                                  reductionIndices);
  }
#endif // defined(BOOST_PP_IS_ITERATING)
