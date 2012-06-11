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
#ifndef __dax_cont_Assert_h
#define __dax_cont_Assert_h

#include <dax/cont/ErrorControlAssert.h>

// Stringify macros for DAX_ASSERT_CONT
#define __DAX_ASSERT_CONT_STRINGIFY_2ND(s) #s
#define __DAX_ASSERT_CONT_STRINGIFY(s) __DAX_ASSERT_CONT_STRINGIFY_2ND(s)

/// \def DAX_ASSERT_CONT(condition)
///
/// Asserts that \a condition resolves to true.  If \a condition is false,
/// then an error is raised.  This macro is meant to work in the Dax control
/// environment and throws an ErrorControlAssert object on failure.

#define DAX_ASSERT_CONT(condition) \
  if (!(condition)) \
    ::dax::cont::Assert(condition, __FILE__, __LINE__, #condition)

namespace dax {
namespace cont {

DAX_CONT_EXPORT void Assert(bool condition,
                            const std::string &file,
                            dax::Id line,
                            const std::string &message)
{
  if (condition)
    {
    // Do nothing.
    }
  else
    {
    throw dax::cont::ErrorControlAssert(file, line, message);
    }
}

}
} // namespace dax::cont

#endif //__dax_cont_Assert_h
