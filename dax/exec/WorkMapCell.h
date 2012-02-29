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

#include <dax/internal/GridTopologys.h>
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
template<class CT> class WorkMapCell
{
public:
  typedef CT CellType;
  typedef typename CellType::TopologyType TopologyType;

private:
  CellType Cell;
  dax::exec::internal::ErrorHandler ErrorHandler;

public:
  DAX_EXEC_EXPORT WorkMapCell(
    const TopologyType &GridTopology,
    const dax::exec::internal::ErrorHandler &errorHandler)
    : Cell(GridTopology, 0),
      ErrorHandler(errorHandler) { }

  DAX_EXEC_EXPORT const CellType GetCell() const
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

  template<typename T>
  DAX_EXEC_EXPORT void GetFieldValues(const dax::exec::FieldPoint<T> &field,
                                          T values[CellType::NUM_POINTS]) const
  {
    dax::Id pointIndices[CellType::NUM_POINTS];
    this->GetCell().GetPointIndices(pointIndices);
    dax::exec::internal::fieldAccessNormalGet<CellType::NUM_POINTS>(field, pointIndices, values);
  }

  DAX_EXEC_EXPORT dax::Vector3 GetFieldValue(
    const dax::exec::FieldCoordinates &, dax::Id vertexIndex) const
  {
    dax::Id pointIndex = this->GetCell().GetPointIndex(vertexIndex);
    const TopologyType &GridTopology = this->GetCell().GetGridTopology();
    return
        dax::exec::internal::fieldAccessUniformCoordinatesGet(GridTopology,
                                                              pointIndex);
  }

  DAX_EXEC_EXPORT void GetFieldValues(
    const dax::exec::FieldCoordinates &,
      dax::Vector3 coords[CellType::NUM_POINTS]) const
  {
    dax::Id indices[CellType::NUM_POINTS];
    this->GetCell().GetPointIndices(indices);

    const TopologyType &gridStructure = this->GetCell().GetGridTopology();
    dax::exec::internal::fieldAccessUniformCoordinatesGet<CellType::NUM_POINTS>(gridStructure, indices, coords);
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
