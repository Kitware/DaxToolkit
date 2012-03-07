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
#ifndef __dax_exec_Assert_h
#define __dax_exec_Assert_h

#include <dax/internal/ExportMacros.h>

// Stringify macros for DAX_ASSERT_EXEC
#define __DAX_ASSERT_EXEC_STRINGIFY_2ND(s) #s
#define __DAX_ASSERT_EXEC_STRINGIFY(s) __DAX_ASSERT_EXEC_STRINGIFY_2ND(s)

/// \def DAX_ASSERT_EXEC(condition, work)
///
/// Asserts that \a condition resolves to true.  If \a condition is false,
/// then an error is raised.  This macro is meant to work in the Dax execution
/// environment and requires the \a work object to raise the error and throw it
/// in the control environment.

#define DAX_ASSERT_EXEC(condition, work) \
  ::dax::exec::Assert( \
      condition, \
      __FILE__ ":" __DAX_ASSERT_EXEC_STRINGIFY(__LINE__) ": " \
      "Assert Failed (" #condition ")", \
      work)

namespace dax {
namespace exec {

/// Implements the assert functionality of DAX_ASSERT_EXEC.
///
template<class WorkType>
DAX_EXEC_EXPORT void Assert(bool condition, const char *message, WorkType work)
{
  if (condition)
    {
    // Do nothing.
    }
  else
    {
    work.RaiseError(message);
    }
}

}
} // namespace dax::exec

#endif //__dax_exec_Assert_h
