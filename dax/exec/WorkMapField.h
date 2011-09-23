/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_exec_WorkMapField_h
#define __dax_exec_WorkMapField_h

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>

#include <dax/internal/GridStructures.h>
#include <dax/exec/internal/FieldAccess.h>

namespace dax { namespace exec {

///----------------------------------------------------------------------------
/// Work for worklets that map fields without regard to topology or any other
/// connectivity information.  The template is because the internal accessors
/// of the cells may change based on type.
template<class CellT>
class WorkMapField
{
  dax::Id Index;

public:
  typedef CellT CellType;

  DAX_EXEC_EXPORT WorkMapField(dax::Id index) : Index(index) { }

  template<typename T>
  DAX_EXEC_EXPORT const T &GetFieldValue(const dax::exec::Field<T> &field) const
  {
    return dax::exec::internal::fieldAccessNormalGet(field, this->GetIndex());
  }

  template<typename T>
  DAX_EXEC_EXPORT void SetFieldValue(dax::exec::Field<T> &field, const T &value)
  {
    dax::exec::internal::fieldAccessNormalSet(field, this->GetIndex(), value);
  }

  DAX_EXEC_EXPORT dax::Id GetIndex() const { return this->Index; }

  DAX_EXEC_EXPORT void SetIndex(dax::Id index) { this->Index = index; }
};

template<>
class WorkMapField<dax::exec::CellVoxel>
{
  const dax::internal::StructureUniformGrid GridStructure;
  dax::Id Index;

public:
  typedef CellVoxel CellType;

  DAX_EXEC_EXPORT WorkMapField(const dax::internal::StructureUniformGrid &gs,
                               dax::Id index)
    : GridStructure(gs), Index(index) { }

  template<typename T>
  DAX_EXEC_EXPORT const T &GetFieldValue(const dax::exec::Field<T> &field) const
  {
    return dax::exec::internal::fieldAccessNormalGet(field, this->GetIndex());
  }

  template<typename T>
  DAX_EXEC_EXPORT void SetFieldValue(dax::exec::Field<T> &field, const T &value)
  {
    dax::exec::internal::fieldAccessNormalSet(field, this->GetIndex(), value);
  }

  DAX_EXEC_EXPORT dax::Vector3 GetFieldValue(
    const dax::exec::FieldCoordinates &)
  {
    // Special case.  Point coordiantes are determined implicitly by index.
    return dax::exec::internal::fieldAccessUniformCoordinatesGet(
          this->GridStructure,
          this->GetIndex());
  }

  DAX_EXEC_EXPORT dax::Id GetIndex() const { return this->Index; }

  DAX_EXEC_EXPORT void SetIndex(dax::Id index) { this->Index = index; }
};

}}

#endif //__dax_exec_WorkMapField_h
