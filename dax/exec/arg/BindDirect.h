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
#include <dax/exec/arg/BindInfo.h>
#include <dax/exec/internal/WorkletBase.h>
#include <boost/utility/enable_if.hpp>

namespace dax { namespace exec { namespace arg {

template <typename Invocation, int N>
class BindDirect
{
  typedef dax::exec::arg::BindInfo<N,Invocation> MyInfo;
  typedef typename MyInfo::AllControlBindings AllControlBindings;
  typedef typename MyInfo::ExecArgType ExecArgType;
  typedef typename MyInfo::Tags Tags;
  ExecArgType ExecArg;
public:
  typedef typename ExecArgType::ReturnType ReturnType;

  DAX_CONT_EXPORT BindDirect(AllControlBindings& bindings):
    ExecArg(dax::exec::arg::GetNthExecArg<N>(bindings)) {}

  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType operator()(const IndexType& id,
                      const dax::exec::internal::WorkletBase& worklet)
    {
    return this->ExecArg(id, worklet);
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(dax::Id id,
                       const dax::exec::internal::WorkletBase& worklet)
    {
    //Look at the concept map traits. If we have the Out tag
    //we know that we must call our ExecArgs SaveExecutionResult.
    //Otherwise we are an input argument and that behavior is undefined
    //and very bad things could happen
    typedef typename Tags::
          template Has<typename dax::cont::sig::Out>::type HasOutTag;
    this->saveResult(id,worklet,HasOutTag());
    }

  //method enabled when we do have the out tag ( or InOut)
  template <typename HasOutTag>
  DAX_EXEC_EXPORT
  void saveResult(int id,
                  const dax::exec::internal::WorkletBase& worklet,
                  HasOutTag,
                  typename boost::enable_if<HasOutTag>::type* = 0)
  {
  this->ExecArg.SaveExecutionResult(id,worklet);
  }

  template <typename HasOutTag>
  DAX_EXEC_EXPORT
  void saveResult(int,
                  const dax::exec::internal::WorkletBase&,
                  HasOutTag,
                  typename boost::disable_if<HasOutTag>::type* = 0)
  {
  }

};

}}} // namespace dax::exec::arg

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax_exec_arg_BindDirect_h
