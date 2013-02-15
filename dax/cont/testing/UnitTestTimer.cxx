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

#include <dax/cont/Timer.h>

#include <dax/cont/internal/testing/Testing.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

void Time()
{
  dax::cont::Timer<> timer;

#ifndef _WIN32
  sleep(1);
#else
  Sleep(1000);
#endif

  dax::Scalar elapsedTime = timer.GetElapsedTime();

  std::cout << "Elapsed time: " << elapsedTime << std::endl;

  DAX_TEST_ASSERT(elapsedTime > 0.999,
                  "Timer did not capture full second wait.");
  DAX_TEST_ASSERT(elapsedTime < 2.0,
                  "Timer counted too far or system really busy.");
}

} // anonymous namespace

int UnitTestTimer(int, char *[])
{
  return dax::cont::internal::Testing::Run(Time);
}
