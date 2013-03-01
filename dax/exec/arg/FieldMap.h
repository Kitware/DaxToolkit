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
#ifndef __dax_exec_arg_FieldMap_h
#define __dax_exec_arg_FieldMap_h

#include <dax/Types.h>
#include <dax/exec/internal/WorkletBase.h>

namespace dax { namespace exec { namespace arg {

/// \headerfile FieldMap.h dax/exec/arg/FieldMap.h
/// \brief Execution environment representation of worklet arguments that
/// is a conversion of an argument by passing the index to another function.
/// this in general is a key value lookup where the key is the current
/// workId and the value is the new id to use when passing to T
template <typename Tags, typename ExecKeyType, typename ExecValueType>
class FieldMap : public ExecValueType
{
  ExecKeyType KeyArg;
public:
  typedef typename ExecValueType::ReturnType ReturnType;
  typedef typename ExecValueType::SaveType SaveType;

  FieldMap(const ExecKeyType& k, const ExecValueType& v):
    ExecValueType(v),
    KeyArg(k)
    {
    }

  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType operator()(const IndexType& index,
                            const dax::exec::internal::WorkletBase& work)
    {
    return ExecValueType::operator()(this->KeyArg(index, work), work);
    }

  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType operator()(const IndexType& index,
                            const dax::exec::internal::WorkletBase& work) const
    {
    return ExecValueType::operator()(this->KeyArg(index, work), work);
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(dax::Id index,
                            const dax::exec::internal::WorkletBase& work) const
    {
    ExecValueType::SaveExecutionResult(this->KeyArg(index,work),work);
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(dax::Id index, const SaveType& v,
                             const dax::exec::internal::WorkletBase& work) const
    {
    ExecValueType::SaveExecutionResult(this->KeyArg(index,work),v,work);
    }
};

}}} // namespace dax::exec::arg

#endif //__dax_exec_arg_FieldMap_h
