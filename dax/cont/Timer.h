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
//=============================================================================
#ifndef __dax_cont_internal_Time_h
#define __dax_cont_internal_Time_h

#include <dax/Types.h>
#include <dax/cont/DeviceAdapter.h>

#ifndef _WIN32
#include <limits.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace dax {
namespace cont {

/// A class that can be used to time operations in Dax that might be occuring
/// in parallel.  You should make sure that the device adapter for the timer
/// matches that being used to execute algorithms to ensure that the thread
/// synchronization is correct.
///
/// The default implementation does not have a guaranteed resolution of the
/// time but should generally be good to about a millisecond.  Device adapter
/// implementations should provide specializations of this class if they have
/// more accurate timers.
///
template<class DeviceAdapter = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class Timer{
public:
  /// When a timer is constructed, all threads are synchronized and the
  /// current time is marked so that GetElapsedTime returns the number of
  /// seconds elapsed since the construction.
  DAX_CONT_EXPORT Timer()
  {
    this->Reset();
  }

  /// Resets the timer. All further calls to GetElapsedTime will report the
  /// number of seconds elapsed since the call to this. This method
  /// synchronizes all asynchronous operations.
  ///
  DAX_CONT_EXPORT void Reset()
  {
    // TODO: Synchronize threads.
    this->StartTime = this->GetCurrentTime();
  }

  /// Returns the elapsed time in seconds between the construction of this
  /// class or the last call to Reset and the time this function is called. The
  /// time returned is measured in wall time. GetElapsedTime may be called any
  /// number of times to get the progressive time. This method synchronizes all
  /// asynchronous operations.
  ///
  DAX_CONT_EXPORT dax::Scalar GetElapsedTime()
  {
    // TODO: Synchronize threads.
    TimeStamp currentTime = this->GetCurrentTime();

    dax::Scalar elapsedTime;
    elapsedTime = currentTime.Seconds - this->StartTime.Seconds;
    elapsedTime += ((currentTime.Microseconds - this->StartTime.Microseconds)
                    /dax::Scalar(1000000));

    return elapsedTime;
  }

private:
  struct TimeStamp {
    dax::internal::Int64Type Seconds;
    dax::internal::Int64Type Microseconds;
  };
  TimeStamp StartTime;

  DAX_CONT_EXPORT TimeStamp GetCurrentTime()
  {
    TimeStamp retval;
#ifdef _WIN32
    timeb currentTime;
    ::ftime(&currentTime);
    retval.Seconds = currentTime.time;
    retval.Microseconds = 1000*currentTime.millitm;
#else
    timeval currentTime;
    gettimeofday(&currentTime, NULL);
    retval.Seconds = currentTime.tv_sec;
    retval.Microseconds = currentTime.tv_usec;
#endif
    return retval;
  }
};

}
} // namespace dax::cont

#endif //__dax_cont_internal_Time_h
