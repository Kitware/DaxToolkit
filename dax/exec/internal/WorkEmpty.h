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
#ifndef __dax_exec_internal_WorkEmpty_h
#define __dax_exec_internal_WorkEmpty_h

#include <dax/internal/ExportMacros.h>

namespace dax {
namespace exec {
namespace internal {

/// This is an empty work object that contains no information about grid
/// structures, data layout, or work division.  It contains just enough
/// about the current scheduling (basically error handling) for internal
/// functions to perform setup/finish operations that are not associated
/// with any particular piece of work.
template<class ExecutionAdapter>
class WorkEmpty
{
private:
  const ExecutionAdapter Adapter;

public:
  DAX_EXEC_CONT_EXPORT WorkEmpty(
      const ExecutionAdapter &executionAdapter)
    : Adapter(executionAdapter) { }

  DAX_EXEC_EXPORT void RaiseError(const char *message) const
  {
    this->Adapter.RaiseError(message);
  }
};

}
}
} // namespace dax::exec::internal

#endif //__dax_exec_internal_WorkEmpty_h
