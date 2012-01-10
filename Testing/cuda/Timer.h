/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __Timer_h
#define __Timer_h

/// This class just wraps around a boost auto_cpu_timer, whos header file
/// does not seem to compile with nvcc.
///
class Timer
{
public:
  Timer();
  ~Timer();

  void restart();
  double elapsed();

private:
  Timer(const Timer &);           // Not implemented
  void operator=(const Timer &);  // Not implemented

  class InternalStruct;
  InternalStruct *Internals;
};

#endif // __Timer_h
