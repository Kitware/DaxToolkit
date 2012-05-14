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

#ifndef __dax_exec_ParametricCoordinates_h
#define __dax_exec_ParametricCoordinates_h

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>
#include <dax/exec/internal/FieldAccess.h>

#include <dax/exec/WorkMapCell.h>
namespace dax {
namespace exec {

//-----------------------------------------------------------------------------
template<class WorkType, class ExecutionAdapter>
DAX_EXEC_EXPORT dax::Vector3 parametricCoordinatesToWorldCoordinates(
  const WorkType &work,
  const dax::exec::CellVoxel &cell,
  const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &coordField,
  const dax::Vector3 pcoords)
{
  dax::Vector3 spacing = cell.GetSpacing();
  dax::Vector3 cellOffset = spacing * pcoords;

  // This is a cheating way to get the coordinate value for index 0.  This is
  // a very special case where you would want just one point coordinate because
  // the rest are implicitly defined.
  dax::Vector3 minCoord = dax::exec::internal::FieldAccess::GetCoordinates(
        coordField, 0, cell.GetGridTopology(), work);

  return cellOffset + minCoord;
}

template<class WorkType, class ExecutionAdapter>
DAX_EXEC_EXPORT dax::Vector3 worldCoordinatesToParametricCoordinates(
  const WorkType &work,
  const dax::exec::CellVoxel &cell,
  const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &coordField,
  const dax::Vector3 wcoords)
{
  // This is a cheating way to get the coordinate value for index 0.  This is
  // a very special case where you would want just one point coordinate because
  // the rest are implicitly defined.
  dax::Vector3 minCoord = dax::exec::internal::FieldAccess::GetCoordinates(
        coordField, 0, cell.GetGridTopology(), work);

  dax::Vector3 cellOffset = wcoords - minCoord;

  dax::Vector3 spacing = cell.GetSpacing();
  return cellOffset / spacing;
}

}
}

#endif //__dax_exec_ParametricCoordinates_h
