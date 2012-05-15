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
#ifndef __dax_exec_internal_TestExecutionAdapter_h
#define __dax_exec_internal_TestExecutionAdapter_h

#include <dax/internal/Testing.h>

struct TestExecutionAdapter
{
  template <typename T>
  struct FieldStructures
  {
    typedef T *IteratorType;
    typedef const T *IteratorConstType;
  };

  class ErrorHandler
  {
  public:
    void RaiseError(const char *message) const
    {
      DAX_TEST_FAIL(message);
    }
  };
};

#endif //__dax_exec_internal_TestExecutionAdapter_h
