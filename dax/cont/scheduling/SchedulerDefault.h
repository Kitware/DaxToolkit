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
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef __dax_cont_scheduling_ScheduleDefault_h
#define __dax_cont_scheduling_ScheduleDefault_h

#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/arg/FieldArrayHandle.h>
#include <dax/cont/arg/FieldConstant.h>
#include <dax/cont/arg/FieldMap.h>
#include <dax/cont/arg/TopologyUniformGrid.h>
#include <dax/cont/arg/TopologyUnstructuredGrid.h>

#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/Bindings.h>

#include <dax/cont/scheduling/CollectCount.h>
#include <dax/cont/scheduling/CreateExecutionResources.h>
#include <dax/cont/scheduling/Scheduler.h>
#include <dax/cont/scheduling/SchedulerTags.h>
#include <dax/cont/scheduling/VerifyUserArgLength.h>

#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/Tag.h>

#include <dax/Types.h>

#include <dax/exec/internal/Functor.h>

#if !(__cplusplus >= 201103L)
# include <dax/internal/ParameterPackCxx03.h>
#endif // !(__cplusplus >= 201103L)

namespace dax { namespace cont { namespace scheduling {

template <class DeviceAdapterTag>
class Scheduler<DeviceAdapterTag,dax::cont::scheduling::ScheduleDefaultTag>
{
public:
  //default constructor so we can insantiate const schedulers
  DAX_CONT_EXPORT Scheduler(){}

#if __cplusplus >= 201103L
  // Note any changes to this method must be reflected in the
  // C++03 implementation.
  template <class WorkletType, typename...T>
  DAX_CONT_EXPORT void Invoke(WorkletType w, T...a) const
    {
    typedef dax::cont::scheduling::VerifyUserArgLength<WorkletType,
              sizeof...(T)> WorkletUserArgs;
    //if you are getting this error you are passing less arguments than requested
    //in the control signature of this worklet
    DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::NotEnoughParameters));

    //if you are getting this error you are passing too many arguments
    //than requested in the control signature of this worklet
    DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::TooManyParameters));

    // Construct the signature of the worklet invocation on the control side.
    typedef WorkletType ControlInvocationSignature(T...);
    typedef typename WorkletType::DomainType DomainType;

    // Bind concrete arguments T...a to the concepts declared in the
    // worklet ControlSignature through ConceptMap specializations.
    // The concept maps also know how to make the arguments available
    // in the execution environment.
    dax::cont::internal::Bindings<ControlInvocationSignature>
      bindings(a...);

    // Visit each bound argument to determine the count to be scheduled.
    dax::Id count=1;
    bindings.ForEachCont(dax::cont::scheduling::CollectCount<DomainType>(count));

    // Visit each bound argument to set up its representation in the
    // execution environment.
    bindings.ForEachCont(
          dax::cont::scheduling::CreateExecutionResources(count));

    // Schedule the worklet invocations in the execution environment.
    dax::exec::internal::Functor<ControlInvocationSignature>
        bindingFunctor(w, bindings);
    dax::cont::internal::DeviceAdapterAlgorithm<DeviceAdapterTag>::
        Schedule(bindingFunctor, count);
    }
#else // !(__cplusplus >= 201103L)
  // For C++03 use Boost.Preprocessor file iteration to simulate
  // parameter packs by enumerating implementations for all argument
  // counts.
#   define BOOST_PP_ITERATION_PARAMS_1 (3, (2, 10, <dax/cont/scheduling/SchedulerDefault.h>))
#   include BOOST_PP_ITERATE()
#endif // !(__cplusplus >= 201103L)
};

} } }

#endif //__dax_cont_Schedule_h

#else // defined(BOOST_PP_IS_ITERATING)
  //we insert the following code where BOOST_PP_ITERATE is at to simulate
  //variadic methods
  // Note any changes to this method must be reflected in the
  // C++11 implementation.
  template <class WorkletType, _dax_pp_typename___T>
  DAX_CONT_EXPORT void Invoke(WorkletType w, _dax_pp_params___(a)) const
    {
    typedef dax::cont::scheduling::VerifyUserArgLength<WorkletType,
                _dax_pp_sizeof___T> WorkletUserArgs;
    //if you are getting this error you are passing less arguments than requested
    //in the control signature of this worklet
    DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::NotEnoughParameters));

    //if you are getting this error you are passing too many arguments
    //than requested in the control signature of this worklet
    DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::TooManyParameters));

    // Construct the signature of the worklet invocation on the control side.
    typedef WorkletType ControlInvocationSignature(_dax_pp_T___);
    typedef typename WorkletType::DomainType DomainType;

    // Bind concrete arguments T...a to the concepts declared in the
    // worklet ControlSignature through ConceptMap specializations.
    // The concept maps also know how to make the arguments available
    // in the execution environment.
    dax::cont::internal::Bindings<ControlInvocationSignature>
      bindings(_dax_pp_args___(a));

    // Visit each bound argument to determine the count to be scheduled.
    dax::Id count=1;
    bindings.ForEachCont(dax::cont::scheduling::CollectCount<DomainType>(count));

    // Visit each bound argument to set up its representation in the
    // execution environment.
    bindings.ForEachCont(
          dax::cont::scheduling::CreateExecutionResources(count));

    // Schedule the worklet invocations in the execution environment.
    dax::exec::internal::Functor<ControlInvocationSignature>
        bindingFunctor(w, bindings);
    dax::cont::internal::DeviceAdapterAlgorithm<DeviceAdapterTag>::
        Schedule(bindingFunctor, count);
    }

#endif // defined(BOOST_PP_IS_ITERATING)
