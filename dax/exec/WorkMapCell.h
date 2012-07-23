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
#ifndef __dax_exec_WorkMapCell_h
#define __dax_exec_WorkMapCell_h

#include <dax/Types.h>
#include <dax/exec/Cell.h>

#include <dax/exec/internal/FieldAccess.h>

namespace dax {
namespace exec {

///----------------------------------------------------------------------------
// Work for worklets that map points to cell. Use this work when the worklets
// need "CellArray" information i.e. information about what points form a cell.
// There are different versions for different cell types, which might have
// different constructors because they identify topology differently.



///----------------------------------------------------------------------------
// Work for worklets that map points to cell. Use this work when the worklets
// need "CellArray" information i.e. information about what points form a cell.
template<class ExecutionAdapter> class WorkMapCell
{
public:
  DAX_EXEC_EXPORT WorkMapCell(
      const ExecutionAdapter &executionAdapter)
    : Adapter(executionAdapter) { }

  DAX_EXEC_EXPORT void RaiseError(const char *message) const
  {
    this->Adapter.RaiseError(message);
  }

private:
  const ExecutionAdapter Adapter;
};


}
}

#endif //__dax_exec_WorkMapCell_h
