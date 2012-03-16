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

#ifndef __dax_cont_internal_ExecutionPackageGrid_h
#define __dax_cont_internal_ExecutionPackageGrid_h

#include <dax/internal/ExportMacros.h>

namespace dax {
namespace cont {
namespace internal {

template<class GridT>
class ExecutionPackageGrid
{
public:
  typedef GridT ControlGridType;
  typedef typename GridT::TopologyType ExecutionGridType;
  typedef typename GridT::CellType ExecutionCellType;

  ExecutionPackageGrid(const ControlGridType &grid)
    : GridTopology(grid.GridTopology) { }

  ExecutionPackageGrid(const ExecutionGridType &grid) : GridTopology(grid) { }

  const ExecutionGridType &GetExecutionObject() const {
    return this->GridTopology;
  }
private:
  ExecutionGridType GridTopology;
};

}
}
}

#endif //__dax_cont_internal_ExecutionPackageGrid_h
