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
#ifndef __dax_cont_scheduling_ScheduleDefault_h
#define __dax_cont_scheduling_ScheduleDefault_h

#include <dax/cont/arg/ImplementedConceptMaps.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/Bindings.h>

#include <dax/internal/ParameterPack.h>

#include <dax/cont/scheduling/CollectCount.h>
#include <dax/cont/scheduling/CreateExecutionResources.h>
#include <dax/cont/scheduling/Scheduler.h>
#include <dax/cont/scheduling/SchedulerTags.h>
#include <dax/cont/scheduling/VerifyUserArgLength.h>

#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/Tag.h>

#include <dax/Types.h>

#include <dax/exec/internal/Functor.h>

namespace dax { namespace cont { namespace scheduling {

template <class DeviceAdapterTag>
class Scheduler<DeviceAdapterTag,dax::cont::scheduling::ScheduleDefaultTag>
{
public:
  //default constructor so we can insantiate const schedulers
  DAX_CONT_EXPORT Scheduler(){}

  template <typename WorkletType, typename Parameters>
  DAX_CONT_EXPORT void Invoke(WorkletType worklet,
                              const Parameters &arguments) const
    {
    typedef dax::cont::scheduling::VerifyUserArgLength<WorkletType,
              Parameters::NUM_PARAMETERS> WorkletUserArgs;
    //if you are getting this error you are passing less arguments than requested
    //in the control signature of this worklet
    DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::NotEnoughParameters));

    //if you are getting this error you are passing too many arguments
    //than requested in the control signature of this worklet
    DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::TooManyParameters));

    // Bind concrete arguments T...a to the concepts declared in the
    // worklet ControlSignature through ConceptMap specializations.
    // The concept maps also know how to make the arguments available
    // in the execution environment.
    typename dax::cont::internal::Bindings<WorkletType, Parameters>::type
        bindings = dax::cont::internal::BindingsCreate(worklet, arguments);

    // Visit each bound argument to determine the count to be scheduled.
    typedef typename WorkletType::DomainType DomainType;
    dax::Id count=1;
    bindings.ForEachCont(
          dax::cont::scheduling::CollectCount<DomainType>(count));

    // Visit each bound argument to set up its representation in the
    // execution environment.
    bindings.ForEachCont(
          dax::cont::scheduling::CreateExecutionResources(count));

    // Schedule the worklet invocations in the execution environment.
    dax::exec::internal::Functor<WorkletType, Parameters>
        bindingFunctor(worklet, bindings);
    dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag>::Schedule(
                                                        bindingFunctor, count);
    }
};

} } } // namespace dax::cont::scheduling

#endif //__dax_cont_Schedule_h
