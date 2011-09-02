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
class WorkMapFieldBase
{
  dax::Id Index;

public:
  __device__ WorkMapFieldBase(dax::Id index)
  {  }

  __device__ dax::Id GetIndex() const { return this->Index; }

  __device__ void SetIndex(dax::Id index) { this->Index = index; }

  template<typename T>
  __device__ const T &GetFieldValue(const dax::exec::Field<T> &field) const
  {
    return dax::exec::internal::fieldAccessNormalGet(field, this->GetIndex());
  }

  template<typename T>
  __device__ void SetFieldValue(dax::exec::Field<T> &field, const T &value)
  {
    dax::exec::internal::fieldAccessNormalSet(field, this->GetIndex(), value);
  }
};

template<class CellT>
class WorkMapField : public dax::exec::WorkMapFieldBase
{
public:
  typedef CellT CellType;

  __device__ WorkMapField(dax::Id index) : WorkMapFieldBase(index) { }
};

template<>
class WorkMapField<CellVoxel> : public dax::exec::WorkMapFieldBase
{
  const dax::internal::StructureUniformGrid &GridStructure;

public:
  typedef CellVoxel CellType;

  __device__ WorkMapField(const dax::internal::StructureUniformGrid &gs,
                          dax::Id index)
    : WorkMapFieldBase(index), GridStructure(gs) { }

  __device__ dax::Vector3 GetFieldValue(const dax::exec::FieldCoordinates &)
  {
    // Special case.  Point coordiantes are determined implicitly by index.
    return dax::exec::internal::fieldAccessUniformCoordinatesGet(
          this->GridStructure,
          this->GetIndex());
  }
};

}}

#endif //__dax_exec_WorkMapField_h
