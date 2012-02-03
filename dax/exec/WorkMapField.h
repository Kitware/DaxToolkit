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

#include <dax/internal/GridTopologys.h>
#include <dax/exec/internal/ErrorHandler.h>
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
  dax::exec::internal::ErrorHandler ErrorHandler;

public:
  typedef CellT CellType;

  DAX_EXEC_CONT_EXPORT WorkMapField(
      const dax::exec::internal::ErrorHandler &errorHandler)
    : ErrorHandler(errorHandler) { }

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

  DAX_EXEC_EXPORT void RaiseError(const char *message)
  {
    this->ErrorHandler.RaiseError(message);
  }
};

template<>
class WorkMapField<dax::exec::CellVoxel>
{
  const dax::internal::TopologyUniform GridTopology;
  dax::Id Index;
  dax::exec::internal::ErrorHandler ErrorHandler;

public:
  typedef CellVoxel CellType;

  DAX_EXEC_CONT_EXPORT WorkMapField(
      const dax::internal::TopologyUniform &gs,
      const dax::exec::internal::ErrorHandler &errorHandler)
    : GridTopology(gs), ErrorHandler(errorHandler) { }

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
          this->GridTopology,
          this->GetIndex());
  }

  DAX_EXEC_EXPORT dax::Id GetIndex() const { return this->Index; }

  DAX_EXEC_EXPORT void SetIndex(dax::Id index) { this->Index = index; }

  DAX_EXEC_EXPORT void RaiseError(const char *message)
  {
    this->ErrorHandler.RaiseError(message);
  }
};

}}

#endif //__dax_exec_WorkMapField_h
