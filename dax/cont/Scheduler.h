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

#ifndef __dax_cont_Scheduler_h
#define __dax_cont_Scheduler_h

#include <dax/Types.h>

#include <dax/cont/scheduling/DetermineScheduler.h>

//include all the specialization of the scheduler class
#include <dax/cont/scheduling/SchedulerDefault.h>
#include <dax/cont/scheduling/SchedulerCells.h>
#include <dax/cont/scheduling/SchedulerGenerateInterpolatedCells.h>
#include <dax/cont/scheduling/SchedulerGenerateKeysValues.h>
#include <dax/cont/scheduling/SchedulerGenerateTopology.h>
#include <dax/cont/scheduling/SchedulerReduceKeysValues.h>
#include <dax/cont/PermutationContainer.h>

#ifndef DAX_USE_VARIADIC_TEMPLATE
# include <dax/internal/ParameterPackCxx03.h>
#endif // !DAX_USE_VARIADIC_TEMPLATE

namespace dax { namespace cont {

template <class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class Scheduler
{
private:
  template <typename WorkletType, typename ParameterPackType>
  DAX_CONT_EXPORT
  void InvokeImpl(WorkletType worklet, const ParameterPackType &arguments) const
  {
    typedef typename dax::cont::scheduling::DetermineScheduler<
                                  WorkletType>::SchedulerTag SchedulerTag;
    typedef dax::cont::scheduling::Scheduler<DeviceAdapterTag,SchedulerTag>
        SchedulerType;
    const SchedulerType realScheduler;
    realScheduler.Invoke(worklet, arguments);
  }
public:
#ifdef DAX_USE_VARIADIC_TEMPLATE
  // Note any changes to this method must be reflected in the
  // C++03 implementation.
  template <class WorkletType, typename...T>
  DAX_CONT_EXPORT void Invoke(WorkletType worklet, T...args) const
    {
    this->InvokeImpl(worklet, dax::internal::make_ParameterPack(args...));
    }
#else // !DAX_USE_VARIADIC_TEMPLATE
  // For C++03 use Boost.Preprocessor file iteration to simulate
  // parameter packs by enumerating implementations for all argument
  // counts.
#     define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 10, <dax/cont/Scheduler.h>))
#     include BOOST_PP_ITERATE()
#endif // !DAX_USE_VARIADIC_TEMPLATE
};

} } //namespace dax::cont

#endif //__dax_cont_Schedule_h

#else // defined(BOOST_PP_IS_ITERATING)
#if _dax_pp_sizeof___T > 0
  template <class WorkletType, _dax_pp_typename___T>
  DAX_CONT_EXPORT void Invoke(WorkletType worklet,
                              _dax_pp_params___(args)) const
    {
    this->InvokeImpl(
          worklet,
          dax::internal::make_ParameterPack(_dax_pp_args___(args)));
    }
#     endif // _dax_pp_sizeof___T > 1
# endif // defined(BOOST_PP_IS_ITERATING)
