/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

#include "Timer.h"

#include <boost/timer/timer.hpp>

//-----------------------------------------------------------------------------
class Timer::InternalStruct
{
public:
  InternalStruct() : timer() { }
  boost::timer::cpu_timer timer;
};

//-----------------------------------------------------------------------------
Timer::Timer() : Internals(new InternalStruct)
{
  this->Internals->timer.start();
}

//-----------------------------------------------------------------------------
Timer::~Timer()
{
  delete this->Internals;
}

//-----------------------------------------------------------------------------
void Timer::restart()
{
  this->Internals->timer.start();
}

//-----------------------------------------------------------------------------
double Timer::elapsed()
{
  return static_cast<double>(this->Internals->timer.elapsed().wall)/1.0e9;
}
