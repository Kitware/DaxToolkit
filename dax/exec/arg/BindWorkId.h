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
#ifndef __dax_exec_arg_BindWorkId_h
#define __dax_exec_arg_BindWorkId_h
#if defined(DAX_DOXYGEN_ONLY)

#else // !defined(DAX_DOXYGEN_ONLY)

#include <dax/Types.h>
#include <dax/cont/internal/Bindings.h>

#include <dax/exec/internal/WorkletBase.h>

namespace dax { namespace exec { namespace arg {

template <typename Invocation>
struct BindWorkId
{
  typedef dax::cont::internal::Bindings<Invocation> AllControlBindings;

  typedef dax::Id ReturnType;

  BindWorkId(AllControlBindings&){}

  DAX_EXEC_EXPORT ReturnType operator()(dax::Id id,
                              const dax::exec::internal::WorkletBase&)
    {
    return id;
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(int,
                 const dax::exec::internal::WorkletBase&) const
    {
    }
};

}}} // namespace dax::exec::arg

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax_exec_arg_BindWorkId_h
