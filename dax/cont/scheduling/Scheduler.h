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
#ifndef __dax_cont_scheduling_Scheduler_h
#define __dax_cont_scheduling_Scheduler_h


namespace dax { namespace cont { namespace scheduling {

template <class DeviceAdapterTag, class SchedulerTag>
class Scheduler
#ifdef DAX_DOXYGEN_ONLY
{
public:
  /// \brief Executes the giving worklet with the user parameters
  /// using the device adapter specified with DeviceAdapterTag
  ///
  template <class WorkletType, typename...T>
  DAX_CONT_EXPORT void Invoke(WorkletType w, T...a) const
    {
    }
};
#else
//this should be an empty class that people specialize
;
#endif

} } }

#endif //__dax_cont_scheduling_Scheduler_h
