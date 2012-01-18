/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_exec_WorkMapReduceCell_h
#define __dax_exec_WorkMapReduceCell_h

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>

#include <dax/internal/GridStructures.h>
#include <dax/exec/internal/FieldAccess.h>

namespace dax {
namespace exec {

///----------------------------------------------------------------------------
/// Work for worklets that map cell information to the map reduce framework
/// The major difference to WorkMapCell is that this offers the ability
/// to write to multiple Field Value places, so that you can do
/// zipped style arrays. This allows the map step to save information
/// to an abritrary number of arrays.
/// There are different versions for different cell types, which might have
/// different constructors because they identify topology differently.
template<class CellType> class WorkMapReduceCell;

template<>
class WorkMapReduceCell<dax::exec::CellVoxel>
{
private:
  dax::exec::CellVoxel Cell;
  dax::Id ArrayLength;

public:
  typedef CellVoxel CellType;

  DAX_EXEC_EXPORT WorkMapReduceCell(
    const dax::internal::StructureUniformGrid &gridStructure,
    dax::Id cellIndex = 0)
    : Cell(gridStructure, cellIndex),
      ArrayLength(dax::internal::numberOfCells(gridStructure))
    { }


  DAX_EXEC_EXPORT const dax::exec::CellVoxel GetCell() const
  {
    return this->Cell;
  }

  template<typename T>
  DAX_EXEC_EXPORT void SetMappedFieldValue(dax::exec::Field<T> &field,
                                           dax::Id mappedIndex,
                                           const T &value)
  {
    dax::Id index = (ArrayLength*mappedIndex)+this->GetCellIndex();
    dax::exec::internal::fieldAccessNormalSet(field,
                                              index,
                                              value);
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
    const dax::internal::StructureUniformGrid &gridStructure
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
};


}
}

#endif //__dax_exec_WorkMapReduceCell_h
