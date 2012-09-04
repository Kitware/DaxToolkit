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
