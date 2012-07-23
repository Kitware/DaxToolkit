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
#ifndef __dax_exec_WorkMapField_h
#define __dax_exec_WorkMapField_h

#include <dax/Types.h>

#include <dax/exec/internal/FieldAccess.h>

namespace dax { namespace exec {

///----------------------------------------------------------------------------
/// Work for worklets that map fields without regard to topology or any other
/// connectivity information.  The template is because the internal accessors
/// of the cells may change based on type.
template<class ExecutionAdapter>
class WorkMapField
{
public:
  DAX_EXEC_CONT_EXPORT WorkMapField(
      const ExecutionAdapter &executionAdapter)
    : Adapter(executionAdapter)
  { }

  DAX_EXEC_EXPORT void RaiseError(const char *message) const
  {
    this->Adapter.RaiseError(message);
  }

private:
  const ExecutionAdapter Adapter;
};

}}

#endif //__dax_exec_WorkMapField_h
