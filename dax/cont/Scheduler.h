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
#include <dax/cont/scheduling/SchedulerGenerateTopology.h>

#if !(__cplusplus >= 201103L)
# include <dax/internal/ParameterPackCxx03.h>
#endif // !(__cplusplus >= 201103L)

namespace dax { namespace cont {

template <class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class Scheduler
{
public:
#if __cplusplus >= 201103L
  // Note any changes to this method must be reflected in the
  // C++03 implementation.
  template <class WorkletType, typename...T>
  DAX_CONT_EXPORT void Invoke(WorkletType w, T...a) const
    {
    typedef typename dax::cont::scheduling::DetermineScheduler<
                                  WorkletType>::SchedulerTag SchedulerTag;
    typedef dax::cont::scheduling::Scheduler<DeviceAdapterTag,SchedulerTag> Scheduler;
    const Scheduler realScheduler;
    realScheduler.Invoke(w,a...);
    }
#else // !(__cplusplus >= 201103L)
  // For C++03 use Boost.Preprocessor file iteration to simulate
  // parameter packs by enumerating implementations for all argument
  // counts.
#     define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 10, <dax/cont/Scheduler.h>))
#     include BOOST_PP_ITERATE()
#endif // !(__cplusplus >= 201103L)
};

} } //namespace dax::cont

#endif //__dax_cont_Schedule_h

#else // defined(BOOST_PP_IS_ITERATING)
#if _dax_pp_sizeof___T > 0
  template <class WorkletType, _dax_pp_typename___T>
  DAX_CONT_EXPORT void Invoke(WorkletType w, _dax_pp_params___(a)) const
    {
    typedef typename dax::cont::scheduling::DetermineScheduler<
                                WorkletType>::SchedulerTag SchedulerTag;
    typedef dax::cont::scheduling::Scheduler<DeviceAdapterTag,SchedulerTag>
        RealScheduler;
    const RealScheduler realScheduler;
    realScheduler.Invoke(w,_dax_pp_args___(a));
    }
#     endif // _dax_pp_sizeof___T > 1
# endif // defined(BOOST_PP_IS_ITERATING)
