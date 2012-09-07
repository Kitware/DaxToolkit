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
#ifndef __dax_exec_arg_FieldConstant_h
#define __dax_exec_arg_FieldConstant_h

/// \namespace dax::exec::arg
/// \brief Execution environment representation of worklet arguments

#include <dax/Types.h>

namespace dax { namespace exec { namespace arg {

/// \headerfile FieldConstant.h dax/exec/arg/FieldConstant.h
/// \brief Execution worklet argument generator for constant Field values.
template <typename T> class FieldConstant
{
  T Value;
public:
  typedef T const& ReturnType;
  FieldConstant(T x): Value(x) {}

  DAX_EXEC_EXPORT ReturnType operator()(dax::Id) const
    {
    return this->Value;
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(int) const
    {
    //empty method, since we don't actually support
    //saving constant field values
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(int, T) const
    {
    //we don't support the api to save to a constant
    //field when mapped to another domain type
    }

};

}}} // namespace dax::exec::arg

#endif //__dax_exec_arg_FieldConstant_h
