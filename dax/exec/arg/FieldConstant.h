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
#include <dax/exec/internal/WorkletBase.h>

namespace dax { namespace exec { namespace arg {

/// \headerfile FieldConstant.h dax/exec/arg/FieldConstant.h
/// \brief Execution worklet argument generator for constant Field values.
template <typename T> class FieldConstant
{
  T Value;
public:
  typedef T const& ReturnType;
  typedef T SaveType;
  typedef T ValueType;
  FieldConstant(T x): Value(x) {}

  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType operator()(const IndexType&,
                      const dax::exec::internal::WorkletBase& ) const
    {
    return this->Value;
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(int,
                 const dax::exec::internal::WorkletBase&) const
    {
    }
};

}}} // namespace dax::exec::arg

#endif //__dax_exec_arg_FieldConstant_h
