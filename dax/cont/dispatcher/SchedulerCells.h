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
#ifndef __dax_cont_dispatcher_SchedulerCells_h
#define __dax_cont_dispatcher_SchedulerCells_h

#include <dax/cont/arg/ImplementedConceptMaps.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/internal/Bindings.h>

#include <dax/internal/ParameterPack.h>

#include <dax/cont/dispatcher/CollectCount.h>
#include <dax/cont/dispatcher/CreateExecutionResources.h>
#include <dax/cont/dispatcher/DetermineIndicesAndGridType.h>
#include <dax/cont/dispatcher/Scheduler.h>
#include <dax/cont/dispatcher/SchedulerTags.h>
#include <dax/cont/dispatcher/VerifyUserArgLength.h>

#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/Tag.h>

#include <dax/Types.h>

#include <dax/exec/internal/Functor.h>

namespace dax { namespace cont { namespace dispatcher {

template <class DeviceAdapterTag>
class Scheduler<DeviceAdapterTag,dax::cont::dispatcher::ScheduleCellsTag>
{
public:
  //default constructor so we can instantiate const schedulers
  DAX_CONT_EXPORT Scheduler(){}

  template <class WorkletType, typename Parameters>
  DAX_CONT_EXPORT void Invoke(WorkletType worklet,
                              const Parameters &arguments) const
    {
    typedef dax::cont::dispatcher::VerifyUserArgLength<WorkletType,
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
    typedef dax::internal::Invocation<WorkletType, Parameters> Invocation;
    typedef typename dax::cont::internal::Bindings<Invocation>::type
        BindingsType;
    BindingsType bindings =
        dax::cont::internal::BindingsCreate(worklet, arguments);

    // Visit each bound argument to determine the count to be scheduled.
    typedef typename WorkletType::DomainType DomainType;
    dax::Id count=1;
    bindings.ForEachCont(dax::cont::dispatcher::CollectCount<DomainType>(count));

    // Visit each bound argument to set up its representation in the
    // execution environment.
    bindings.ForEachCont(
          dax::cont::dispatcher::CreateExecutionResources(count));

    //if the grid type matches what we are looking for, lets pull
    //out the new count object and use that.
    //we have the bind
    typedef typename dax::cont::dispatcher::DetermineIndicesAndGridType<
                            Invocation>  CellSchedulingIndices;

    typedef typename CellSchedulingIndices::GridTypeTag GridTypeTag;

    dax::exec::internal::Functor<Invocation> bindingFunctor(worklet, bindings);

    CellSchedulingIndices cellScheduler(bindings,count);

    if(cellScheduler.isValidForGridScheduling())
      {
      // Schedule the worklet invocations in the execution environment.
      dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag>::Schedule(
                          bindingFunctor,
                          cellScheduler.gridCount());
      }
    else
      {
      // Schedule the worklet invocations in the execution environment.
      dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag>::Schedule(
                                                    bindingFunctor,
                                                    count);
      }
    }
};

} } }

#endif //__dax_cont_Schedule_h
