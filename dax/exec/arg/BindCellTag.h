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
#ifndef __dax_exec_arg_BindCellTag_h
#define __dax_exec_arg_BindCellTag_h

#include <dax/Types.h>
#include <dax/exec/arg/BindInfo.h>
#include <dax/exec/internal/WorkletBase.h>

namespace dax { namespace exec { namespace arg {

template <typename WorkletType, typename ControlInvocationParams, int N>
class BindCellTag
{
public:
  typedef dax::exec::arg::BindInfo<
      N,WorkletType,ControlInvocationParams> MyInfo;
  typedef typename MyInfo::AllControlBindings AllControlBindings;
  typedef typename MyInfo::ExecArgType ExecArgType;

  // BindCellTag is an empty class so we don't need to worry about the return type
  // based on in or out (although for other derived units like the point ids,
  // it matters substantially).
  typedef typename ExecArgType::CellTag const& ReturnType;
  typedef typename ExecArgType::CellTag SaveType;

   DAX_CONT_EXPORT BindCellTag(const AllControlBindings& daxNotUsed(bindings)):
    CellTag()
    {}

  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType operator()(
      const IndexType& daxNotUsed(index),
      const dax::exec::internal::WorkletBase& daxNotUsed(work))
  {
    return this->CellTag;
  }

  DAX_EXEC_EXPORT void SaveExecutionResult(int daxNotUsed(index),
    const dax::exec::internal::WorkletBase& daxNotUsed(work)) const
    {
    //nothing to save
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(int daxNotUsed(index),
    const SaveType& daxNotUsed(v),
    const dax::exec::internal::WorkletBase& daxNotUsed(work)) const
    {
    //nothing to save
    }
private:
  SaveType CellTag;
};


} } } //namespace dax::exec::arg



#endif //__dax_exec_arg_BindCellTag_h
