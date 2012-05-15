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
#include <dax/exec/Field.h>

#include <dax/exec/internal/ErrorHandler.h>
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
template<class CT, class ExecutionAdapter> class WorkMapCell
{
public:
  typedef CT CellType;
  typedef typename CellType::template GridStructures<ExecutionAdapter>
      ::TopologyType TopologyType;

private:
  const CellType Cell;
  const dax::exec::internal::ErrorHandler<ExecutionAdapter> ErrorHandler;

public:
  DAX_EXEC_EXPORT WorkMapCell(
      const TopologyType &GridTopology,
      dax::Id index,
      const dax::exec::internal::ErrorHandler<ExecutionAdapter> &errorHandler)
    : Cell(GridTopology, index),
      ErrorHandler(errorHandler) { }

  DAX_EXEC_EXPORT const CellType GetCell() const
  {
    return this->Cell;
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
  DAX_EXEC_EXPORT void SetFieldValue(
      dax::exec::FieldCellOut<T, ExecutionAdapter> field,
      T value) const
  {
    dax::exec::internal::FieldAccess::SetNormal(field,
                                                this->GetCellIndex(),
                                                value,
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

  DAX_EXEC_EXPORT void RaiseError(const char *message) const
  {
    this->ErrorHandler.RaiseError(message);
  }
};


}
}

#endif //__dax_exec_WorkMapCell_h
