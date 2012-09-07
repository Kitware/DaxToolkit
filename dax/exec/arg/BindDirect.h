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
#ifndef __dax_exec_arg_BindDirect_h
#define __dax_exec_arg_BindDirect_h
#if defined(DAX_DOXYGEN_ONLY)

#else // !defined(DAX_DOXYGEN_ONLY)

#include <dax/Types.h>
#include <dax/cont/internal/Bindings.h>

namespace dax { namespace exec { namespace arg {

template <typename Invocation, int N>
struct BindDirect
{
  typedef dax::cont::internal::Bindings<Invocation> ControlBindings;
  typedef typename ControlBindings::template GetType<N>::type ControlBinding;
  typedef typename ControlBinding::ExecArg ExecArgType;
  ExecArgType ExecArg;

  typedef typename ExecArgType::ReturnType ReturnType;

  BindDirect(ControlBindings& bindings):
    ExecArg(bindings.template Get<N>().GetExecArg()) {}

  DAX_EXEC_EXPORT ReturnType operator()(dax::Id id)
    {
    return this->ExecArg(id);
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(int id)
    {
    this->ExecArg.SaveExecutionResult(id);
    }

};

}}} // namespace dax::exec::arg

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax_exec_arg_BindDirect_h
