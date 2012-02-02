/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_exec_WorkMapCell_h
#define __dax_exec_WorkMapCell_h

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>

#include <dax/internal/GridStructures.h>
#include <dax/exec/internal/ErrorHandler.h>
#include <dax/exec/internal/FieldAccess.h>

namespace dax {
namespace exec {

///----------------------------------------------------------------------------
// Work for worklets that map points to cell. Use this work when the worklets
// need "CellArray" information i.e. information about what points form a cell.
// There are different versions for different cell types, which might have
// different constructors because they identify topology differently.
template<class CellType> class WorkMapCell;

template<>
class WorkMapCell<dax::exec::CellVoxel>
{
private:
  dax::exec::CellVoxel Cell;
  dax::exec::internal::ErrorHandler ErrorHandler;

public:
  typedef CellVoxel CellType;

  DAX_EXEC_EXPORT WorkMapCell(
      const dax::internal::TopologyUniformGrid &gridStructure,
      const dax::exec::internal::ErrorHandler &errorHandler)
    : Cell(gridStructure, 0), ErrorHandler(errorHandler) { }

  DAX_EXEC_EXPORT const dax::exec::CellVoxel GetCell() const
  {
    return this->Cell;
  }

  template<typename T>
  DAX_EXEC_EXPORT
  const T &GetFieldValue(const dax::exec::FieldCell<T> &field) const
  {
    return dax::exec::internal::fieldAccessNormalGet(field,
                                                     this->GetCellIndex());
  }

  template<typename T>
  DAX_EXEC_EXPORT void SetFieldValue(dax::exec::FieldCell<T> &field,
                                     const T &value)
  {
    dax::exec::internal::fieldAccessNormalSet(field,
                                              this->GetCellIndex(),
                                              value);
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
    const dax::internal::TopologyUniformGrid &gridStructure
        = this->GetCell().GetGridStructure();
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

///----------------------------------------------------------------------------
// Work for worklets that map points to cell. Use this work when the worklets
// need "CellArray" information i.e. information about what points form a cell.
template<>
class WorkMapCell<dax::exec::CellHexahedron>
{
private:
  dax::exec::CellHexahedron Cell;
  dax::exec::internal::ErrorHandler ErrorHandler;

public:
  typedef CellHexahedron CellType;

  DAX_EXEC_EXPORT WorkMapCell(
    const dax::internal::TopologyUnstructuredGrid<CellType> &gridStructure,
    const dax::exec::internal::ErrorHandler &errorHandler)
    : Cell(gridStructure, 0),
      ErrorHandler(errorHandler) { }

  DAX_EXEC_EXPORT const dax::exec::CellHexahedron GetCell() const
  {
    return this->Cell;
  }

  template<typename T>
  DAX_EXEC_EXPORT
  const T &GetFieldValue(const dax::exec::FieldCell<T> &field) const
  {
    return dax::exec::internal::fieldAccessNormalGet(field,
                                                     this->GetCellIndex());
  }

  template<typename T>
  DAX_EXEC_EXPORT void SetFieldValue(dax::exec::FieldCell<T> &field,
                                     const T &value)
  {
    dax::exec::internal::fieldAccessNormalSet(field,
                                              this->GetCellIndex(),
                                              value);
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
    const dax::internal::TopologyUnstructuredGrid<CellType> &gridStructure
        = this->GetCell().GetGridStructure();
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

#endif //__dax_exec_WorkMapCell_h
