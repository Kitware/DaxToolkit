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

namespace detail {

template<class GridType>
DAX_CONT_EXPORT static typename GridType::ExecutionTopologyStruct
ExecutionPackageGridInternal(const GridType &grid,
                             dax::cont::UnstructuredGridTag)
{
  typename GridType::ExecutionTopologyStruct topology;
  topology.CellConnections
      = grid.GetCellConnections().PrepareForInput().first;
  topology.NumberOfPoints = grid.GetNumberOfPoints();
  topology.NumberOfCells = grid.GetNumberOfCells();
  return topology;
}

template<class GridType>
DAX_CONT_EXPORT static typename dax::exec::internal::TopologyUniform
ExecutionPackageGridInternal(const GridType &grid, dax::cont::UniformGridTag)
{
  dax::exec::internal::TopologyUniform topology;
  topology.Origin = grid.GetOrigin();
  topology.Spacing = grid.GetSpacing();
  topology.Extent = grid.GetExtent();
  return topology;
}

} // namespace detail

template<class GridType>
DAX_CONT_EXPORT static typename GridType::ExecutionTopologyStruct
ExecutionPackageGrid(const GridType &grid)
{
  return detail::ExecutionPackageGridInternal(grid,
                                              typename GridType::GridTypeTag());
}

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_ExecutionPackageGrid_h
