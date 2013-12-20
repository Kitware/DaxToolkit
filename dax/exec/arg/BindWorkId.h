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
  //can't used BindInfo, since the WorkId has no control side mapping
  //it is only an exec argument
  typedef typename dax::cont::internal::Bindings<Invocation>::type
      AllControlBindings;

  typedef dax::Id ReturnType;

  DAX_CONT_EXPORT BindWorkId(AllControlBindings& daxNotUsed(bindings)) {}

  // Explicitly making this copy constructor avoids a warning.  For some
  // reason the default implementation with the gcc compiler somehow uses
  // an uninitalized value.  I don't know how since there are no ivars in
  // this class, but this seems to solve the problem.
  DAX_EXEC_CONT_EXPORT BindWorkId(const BindWorkId &) {}

  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType operator()(const IndexType& id,
                              const dax::exec::internal::WorkletBase&)
    {
    return id.value();
    }

  DAX_EXEC_EXPORT ReturnType operator()(const dax::Id& id,
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
