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

#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

#include <dax/exec/internal/GridTopologies.h>

namespace dax {
namespace cont {
namespace internal {

class ExecutionPackageGrid
{
private:
  template<class GridType>
  DAX_CONT_EXPORT static typename GridType::ExecutionTopologyStruct
  GetExecutionObject(const GridType &grid, dax::cont::UnstructuredGridTag)
  {
    typename GridType::ExecutionTopologyStruct topology;
    topology.PointCoordinates
        = grid.GetPointCoordinates().PrepareForInput().first;
    topology.NumberOfPoints = grid.GetNumberOfPoints();
    topology.CellConnections
        = grid.GetCellConnections().PrepareForInput().first;
    topology.NumberOfCells = grid.GetNumberOfCells();
    return topology;
  }

  template<class GridType>
  DAX_CONT_EXPORT static typename dax::exec::internal::TopologyUniform
  GetExecutionObject(const GridType &grid, dax::cont::UniformGridTag)
  {
    dax::exec::internal::TopologyUniform topology;
    topology.Origin = grid.GetOrigin();
    topology.Spacing = grid.GetSpacing();
    topology.Extent = grid.GetExtent();
    return topology;
  }

public:
  template<class GridType>
  DAX_CONT_EXPORT static typename GridType::ExecutionTopologyStruct
  GetExecutionObject(const GridType &grid)
  {
    return GetExecutionObject(grid, typename GridType::GridTypeTag());
  }
};

}
}
}

#endif //__dax_cont_internal_ExecutionPackageGrid_h
