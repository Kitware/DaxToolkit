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
//  Copyright 2013 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_exec_arg_BindPermutedCellField_h
#define __dax_exec_arg_BindPermutedCellField_h
#if defined(DAX_DOXYGEN_ONLY)

#else // !defined(DAX_DOXYGEN_ONLY)

#include <dax/Types.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/arg/Topology.h>
#include <dax/cont/internal/Bindings.h>
#include <dax/cont/internal/FindBinding.h>
#include <dax/cont/sig/Tag.h>

#include <dax/exec/CellVertices.h>
#include <dax/exec/internal/IJKIndex.h>

#include <dax/exec/internal/WorkletBase.h>

#include <boost/utility/enable_if.hpp>

namespace dax { namespace exec { namespace arg {

template <typename Invocation, int N>
class BindPermutedCellField
{
  typedef typename dax::cont::internal::Bindings<Invocation> BindingsType;
  typedef typename dax::cont::internal::FindBinding<
      BindingsType, dax::cont::arg::Topology>::type TopoIndex;
  typedef typename BindingsType::template GetType<TopoIndex::value>::type
      TopoControlBinding;
  typedef typename TopoControlBinding::ExecArg TopoExecArgType;
  TopoExecArgType TopoExecArg;

  typedef typename dax::cont::internal::Bindings<Invocation>
      ::template GetType<N>::type ControlBinding;
  typedef typename ControlBinding::ExecArg ExecArgType;
  typedef typename dax::cont::arg::ConceptMapTraits<ControlBinding>::Tags Tags;
  ExecArgType ExecArg;

public:
  typedef typename ExecArgType::ValueType ValueType;
  ValueType Value;

  typedef typename boost::mpl::if_<
      typename Tags::template Has<dax::cont::sig::Out>,
      ValueType&,
      ValueType const&>::type ReturnType;
  typedef ValueType SaveType;

  DAX_CONT_EXPORT BindPermutedCellField(
      dax::cont::internal::Bindings<Invocation>& bindings):
    TopoExecArg(bindings.template Get<TopoIndex::value>().GetExecArg()),
    ExecArg(bindings.template Get<N>().GetExecArg()),
    Value() {}

  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType operator()(
      const IndexType& workletIndex,
      const dax::exec::internal::WorkletBase& work)
    {
    dax::Id cellIndex = this->TopoExecArg.GetMapIndex(workletIndex, work);
    this->Value = this->ExecArg(cellIndex, work);
    return this->Value;
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(
      int workletIndex,
      const dax::exec::internal::WorkletBase& worklet) const
    {
    //Look at the concept map traits. If we have the Out tag
    //we know that we must call our ExecArgs SaveExecutionResult.
    //Otherwise we are an input argument and that behavior is undefined
    //and very bad things could happen
    typedef typename Tags::
          template Has<typename dax::cont::sig::Out>::type HasOutTag;
    this->saveResult(workletIndex,worklet,HasOutTag());
    }

  //method enabled when we do have the out tag ( or InOut)
  template <typename HasOutTag>
  DAX_EXEC_EXPORT
  void saveResult(int workletIndex,
                  const dax::exec::internal::WorkletBase& work,
                  HasOutTag,
                  typename boost::enable_if<HasOutTag>::type* = 0) const
    {
    dax::Id cellIndex = this->TopoExecArg.GetMapIndex(workletIndex, work);
    this->ExecArg.SaveExecutionResult(cellIndex, this->Value, work);
    }

  template <typename HasOutTag>
  DAX_EXEC_EXPORT
  void saveResult(int,
                  const dax::exec::internal::WorkletBase&,
                  HasOutTag,
                  typename boost::disable_if<HasOutTag>::type* = 0) const
    {
    }
};

}}} // namespace dax::exec::arg

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax_exec_arg_BindPermutedCellField_h
