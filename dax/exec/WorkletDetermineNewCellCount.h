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
#ifndef __dax_exec_WorkletDetermineNewCellCount_h
#define __dax_exec_WorkletDetermineNewCellCount_h

#include <dax/exec/WorkletMapCell.h>

namespace dax {
namespace exec {

///----------------------------------------------------------------------------
/// Worklet that determines how many new cells should be generated
/// from it with the same topology.
/// This worklet is based on the WorkletMapCell type so you have access to
/// "CellArray" information i.e. information about what points form a cell.
/// There are different versions for different cell types, which might have
/// different constructors because they identify topology differently.

class WorkletDetermineNewCellCount : public dax::exec::WorkletMapCell
{
public:

  DAX_EXEC_EXPORT WorkletDetermineNewCellCount() { }
};


}
}

#endif //__dax_exec_WorkletDetermineNewCellCount_h
