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

  this->InvokeGenerateKeysValues(w,T...);
  }

  //todo implement the InvokeGenerateKeysValues method with C11 syntax

#else
# define BOOST_PP_ITERATION_PARAMS_1 (3, (2, 10, <dax/cont/scheduling/SchedulerGenerateKeysValues.h>))
# include BOOST_PP_ITERATE()
#endif

private:

//want the basic implementation to be easily edited, instead of inside
//the BOOST_PP block and unreadable. This version of InvokeGenerateKeysValues
//handles the use case no parameters
template <class WorkletType,
          typename ClassifyHandleType,
          typename InputGrid>
DAX_CONT_EXPORT void InvokeGenerateKeysValues(
    dax::cont::GenerateKeysValues<
    WorkletType,ClassifyHandleType>& workletWrapper,
    const InputGrid& inputGrid) const
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
      dax::cont::make_Permutation(outputIndexRanges,inputGrid,
                                  inputGrid.GetNumberOfCells()),
      visitIndex);
  }

  typedef dax::cont::scheduling::Scheduler<DeviceAdapterTag,
    dax::cont::scheduling::ScheduleDefaultTag> SchedulerDefaultType;
  const SchedulerDefaultType DefaultScheduler;

};

} } }

#endif //__dax_cont_scheduling_GenerateKeysValues_h

#else // defined(BOOST_PP_IS_ITERATING)
public: //needed so that each iteration of invoke is public
template <class WorkletType, _dax_pp_typename___T>
DAX_CONT_EXPORT void Invoke(WorkletType w, _dax_pp_params___(a)) const
  {
  //we are being passed dax::cont::GenerateKeysValues,
  //we want the actual exec worklet that is being passed to scheduleGenerateTopo
  typedef typename WorkletType::WorkletType RealWorkletType;
  typedef dax::cont::scheduling::VerifyUserArgLength<RealWorkletType,
              _dax_pp_sizeof___T> WorkletUserArgs;
  //if you are getting this error you are passing less arguments than requested
  //in the control signature of this worklet
  DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::NotEnoughParameters));

  //if you are getting this error you are passing too many arguments
  //than requested in the control signature of this worklet
  DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::TooManyParameters));

  this->InvokeGenerateKeysValues(w,_dax_pp_args___(a));
  }

private:
template <class WorkletType,
          typename ClassifyHandleType,
          typename InputGrid,
          _dax_pp_typename___T>
DAX_CONT_EXPORT void InvokeGenerateKeysValues(
    dax::cont::GenerateKeysValues<
    WorkletType, ClassifyHandleType >& workletWrapper,
    const InputGrid& inputGrid,
    _dax_pp_params___(a)) const
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

  //we get our magic here. we need to wrap some parameters and pass
  //them to the real scheduler. The visitIndex must be last, as that is the
  //hardcoded location the ReplaceAndExtendSignatures will place it at
  this->DefaultScheduler.Invoke(
      derivedWorklet,
      dax::cont::make_Permutation(outputIndexRanges,inputGrid,
                                  inputGrid.GetNumberOfCells()),
      _dax_pp_args___(a),
      visitIndex);
  }
#endif // defined(BOOST_PP_IS_ITERATING)
