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
#ifndef __dax_exec_WorkMapField_h
#define __dax_exec_WorkMapField_h

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>

#include <dax/exec/internal/ErrorHandler.h>
#include <dax/exec/internal/FieldAccess.h>

namespace dax { namespace exec {

///----------------------------------------------------------------------------
/// Work for worklets that map fields without regard to topology or any other
/// connectivity information.  The template is because the internal accessors
/// of the cells may change based on type.
template<class CellT, class ExecutionAdapter>
class WorkMapField
{
public:
  typedef CellT CellType;
  typedef typename CellType::template GridStructures<ExecutionAdapter>
      ::TopologyType TopologyType;

private:
  const TopologyType GridTopology;
  const dax::Id Index;
  const dax::exec::internal::ErrorHandler<ExecutionAdapter> ErrorHandler;

public:
  DAX_EXEC_CONT_EXPORT WorkMapField(
      const TopologyType &gs,
      dax::Id index,
      const dax::exec::internal::ErrorHandler<ExecutionAdapter> &errorHandler)
    : GridTopology(gs), Index(index), ErrorHandler(errorHandler)
  { }

  template<typename T, template<typename, class> class Access, class Assoc>
  DAX_EXEC_EXPORT T GetFieldValue(
      dax::exec::internal::FieldBase<
          Access<T,ExecutionAdapter>, Assoc> field) const
  {
    return dax::exec::internal::FieldAccess::GetNormal(field,
                                                       this->GetIndex(),
                                                       *this);
  }

  template<typename T, class Assoc>
  DAX_EXEC_EXPORT void SetFieldValue(
      dax::exec::internal::FieldBase<
          dax::exec::internal::FieldAccessPolicyOutput<T, ExecutionAdapter>,
          Assoc> field,
      T value) const
  {
    dax::exec::internal::FieldAccess::SetNormal(field,
                                                this->GetIndex(),
                                                value,
                                                *this);
  }

  DAX_EXEC_EXPORT dax::Vector3 GetFieldValue(
      dax::exec::internal::FieldBase<
          dax::exec::internal::FieldAccessPolicyInput<dax::Vector3, ExecutionAdapter>,
          dax::exec::internal::FieldAssociationCoordinatesTag> field) const
  {
    // Special case.  Point coordiantes can bedetermined implicitly by index.
    return dax::exec::internal::FieldAccess::GetCoordinates(
          field, this->GetIndex(), this->GridTopology, *this);
  }

  DAX_EXEC_EXPORT dax::Id GetIndex() const { return this->Index; }

  DAX_EXEC_EXPORT void RaiseError(const char *message) const
  {
    this->ErrorHandler.RaiseError(message);
  }
};

}}

#endif //__dax_exec_WorkMapField_h
