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
#ifndef __dax_exec_WorkDetermineNewCellCount_h
#define __dax_exec_WorkDetermineNewCellCount_h

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapCell.h>

#include <dax/exec/internal/ErrorHandler.h>
#include <dax/exec/internal/FieldAccess.h>

namespace dax {
namespace exec {

///----------------------------------------------------------------------------
/// Worklet that determines how many new cells should be generated
/// from it with the same topology.
/// This worklet is based on the WorkMapCell type so you have access to
/// "CellArray" information i.e. information about what points form a cell.
/// There are different versions for different cell types, which might have
/// different constructors because they identify topology differently.

template<class CT, class ExecutionAdapter>
class WorkDetermineNewCellCount
{
public:
  typedef CT CellType;
  typedef typename CellType::TopologyType TopologyType;

private:
  CellType Cell;
  dax::exec::FieldCellOut<dax::Id, ExecutionAdapter> NewCellCount;
  dax::exec::internal::ErrorHandler ErrorHandler;
public:

  DAX_EXEC_EXPORT WorkDetermineNewCellCount(
    const TopologyType &gridStructure,
    const dax::exec::FieldCellOut<dax::Id, ExecutionAdapter> &cellCount,
    const dax::exec::internal::ErrorHandler &errorHandler)
    : Cell(gridStructure, 0),
      NewCellCount(cellCount),
      ErrorHandler(errorHandler)
    { }

  DAX_EXEC_EXPORT const CellType GetCell() const
  {
    return this->Cell;
  }

  //Set the number of cells you want this cell to generate
  DAX_EXEC_EXPORT void SetNewCellCount(dax::Id value)
  {
    dax::exec::internal::FieldAccess::SetNormal(this->NewCellCount,
                                                this->GetCellIndex(),
                                                value);
  }

  template<typename T, template<typename, class> class Access>
  DAX_EXEC_EXPORT T GetFieldValue(
      dax::exec::internal::FieldBase<
          Access<T,ExecutionAdapter>,
          dax::exec::internal::FieldAssociationCellTag> field) const
  {
    return dax::exec::internal::FieldAccess::GetNormal(field,
                                                       this->GetCellIndex(),
                                                       *this);
  }

  template<typename T>
  DAX_EXEC_EXPORT dax::Tuple<T,CellType::NUM_POINTS> GetFieldValues(
      dax::exec::FieldPointIn<T, ExecutionAdapter> field) const
  {
    return dax::exec::internal::FieldAccess::GetMultiple(
          field, this->GetCell().GetPointIndices(), *this);
  }

  DAX_EXEC_EXPORT
  dax::Tuple<dax::Vector3,CellType::NUM_POINTS> GetFieldValues(
      dax::exec::FieldCoordinatesIn<ExecutionAdapter> field) const
  {
    return dax::exec::internal::FieldAccess::GetCoordinatesMultiple(
          field,
          this->GetCell().GetPointIndices(),
          this->GetCell().GetGridTopology(),
          *this);
  }

  DAX_EXEC_EXPORT dax::Id GetCellIndex() const { return this->Cell.GetIndex(); }

  DAX_EXEC_EXPORT void SetCellIndex(dax::Id cellIndex)
  {
    this->Cell.SetIndex(cellIndex);
  }

  DAX_EXEC_EXPORT void RaiseError(const char* message)
  {
    this->ErrorHandler.RaiseError(message);
  }
};


}
}

#endif //__dax_exec_WorkDetermineNewCellCount_h
