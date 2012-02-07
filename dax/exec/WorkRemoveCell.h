/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_exec_WorkRemoveCell_h
#define __dax_exec_WorkRemoveCell_h

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapCell.h>

#include <dax/internal/GridTopologys.h>
#include <dax/exec/internal/ErrorHandler.h>
#include <dax/exec/internal/FieldAccess.h>

namespace dax {
namespace exec {

///----------------------------------------------------------------------------
// Work for worklets that determine if a cell should be removed. This
// worklet is based on the WorkMapCell type so you have access to
// "CellArray" information i.e. information about what points form a cell.
// There are different versions for different cell types, which might have
// different constructors because they identify topology differently.

template<class CellType> class WorkRemoveCell;


template<>
class WorkRemoveCell<dax::exec::CellVoxel>
{
public:
  typedef dax::exec::CellVoxel CellType;

private:
  CellType Cell;
  dax::exec::internal::ErrorHandler ErrorHandler;
  dax::exec::FieldCell<dax::Id> RemoveCell;
  
  DAX_EXEC_EXPORT WorkRemoveCell(
    const dax::internal::TopologyUniform &gridStructure,
    const dax::exec::internal::ErrorHandler &errorHandler,
    const dax::exec::FieldCell<dax::Id> &removeCell,
    dax::Id cellIndex = 0)
    : Cell(gridStructure, cellIndex),
      ErrorHandler(errorHandler),
      RemoveCell(removeCell)
    { }

  DAX_EXEC_EXPORT const dax::exec::CellVoxel GetCell() const
  {
    return this->Cell;
  }

  //set this to true if you want to remove this cell
  //Any cell with the value of zero is removed.
  DAX_EXEC_EXPORT void SetRemoveCell(dax::Id value)
  {
    dax::exec::internal::fieldAccessNormalSet(this->RemoveCell,
                                              this->GetCellIndex(),
                                              value);
  }

  //set this to true if you want to remove this cell
  DAX_EXEC_EXPORT dax::Id IsCellRemoved()
  {
    return dax::exec::internal::fieldAccessNormalGet(this->RemoveCell,
                                              this->GetCellIndex());
  }

  template<typename T>
  DAX_EXEC_EXPORT
  const T &GetFieldValue(const dax::exec::FieldCell<T> &field) const
  {
    return dax::exec::internal::fieldAccessNormalGet(field,
                                                     this->GetCellIndex());
  }

  template<typename T>
  DAX_EXEC_EXPORT const T &GetFieldValue(const dax::exec::FieldPoint<T> &field,
                                         dax::Id vertexIndex) const
  {
    dax::Id pointIndex = this->GetCell().GetPointIndex(vertexIndex);
    return dax::exec::internal::fieldAccessNormalGet(field, pointIndex);
  }

  DAX_EXEC_EXPORT dax::Vector3 GetFieldValue(
    const dax::exec::FieldCoordinates &, dax::Id vertexIndex) const
  {
    dax::Id pointIndex = this->GetCell().GetPointIndex(vertexIndex);
    const dax::internal::TopologyUniform &gridStructure
        = this->GetCell().GetGridTopology();
    return
        dax::exec::internal::fieldAccessUniformCoordinatesGet(gridStructure,
                                                              pointIndex);
  }

  DAX_EXEC_EXPORT dax::Id GetCellIndex() const { return this->Cell.GetIndex(); }

  DAX_EXEC_EXPORT void SetCellIndex(dax::Id cellIndex)
  {
    this->Cell.SetIndex(cellIndex);
  }

  DAX_EXEC_EXPORT void RaiseError(const char *message)
  {
    this->ErrorHandler.RaiseError(message);
  }  
};


}
}

#endif //__dax_exec_WorkRemoveCell_h
