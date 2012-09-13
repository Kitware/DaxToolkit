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

namespace dax { namespace exec { namespace arg {

/// \headerfile FieldMap.h dax/exec/arg/FieldMap.h
/// \brief Execution environment representation of worklet arguments that
/// is a conversion of an argument by passing the index to another function.
/// this in general is a key value lookup where the key is the current
/// workId and the value is the new id to use when passing to T
template <typename Tags, typename Value, typename Key> class FieldMap
{
private:
  typedef Value ExecValueType;
  typedef Key ExecKeyType;
  ExecValueType& ValueArg;
  ExecKeyType& KeyArg;
public:
  typedef typename ExecValueType::ReferenceType ReferenceType;
  typedef typename ExecValueType::ReturnType ReturnType;

  FieldMap(){}

  DAX_CONT_EXPORT void SetValueExecArg(ExecValueType& arg)
    {
    this->ValueArg = arg;
    }

  DAX_CONT_EXPORT void SetKeyExecArg(ExecKeyType& arg)
    {
    this->KeyArg = arg;
    }

  template< typename Worklet>
  DAX_EXEC_EXPORT ReturnType operator()(dax::Id index, const Worklet& work)
    {
    return this->ValueArg(this->KeyArg(index, work), work);
    }

  template< typename Worklet>
  DAX_EXEC_EXPORT void SaveExecutionResult(int index, const Worklet& work) const
    {
    this->ValueArg.SaveExecutionResult(this->KeyArg(index,work),work);

    }

  template< typename Worklet>
  DAX_EXEC_EXPORT void SaveExecutionResult(int index, ReferenceType v,
                                           const Worklet& work) const
    {
    this->ValueArg.SaveExecutionResult(this->KeyArg(index,work),v,work);
    }
};

}}} // namespace dax::exec::arg

#endif //__dax_exec_arg_FieldMap_h
