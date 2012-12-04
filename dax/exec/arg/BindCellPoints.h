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
#ifndef __dax_exec_arg_BindCellPoints_h
#define __dax_exec_arg_BindCellPoints_h
#if defined(DAX_DOXYGEN_ONLY)

#else // !defined(DAX_DOXYGEN_ONLY)

#include <dax/Types.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/arg/Topology.h>
#include <dax/cont/internal/Bindings.h>
#include <dax/cont/internal/FindBinding.h>
#include <dax/cont/sig/Tag.h>

#include <dax/exec/CellField.h>
#include <dax/exec/CellVertices.h>

#include <dax/exec/internal/WorkletBase.h>

#include <boost/utility/enable_if.hpp>

namespace dax { namespace exec { namespace arg {

template <typename Invocation, int N>
class BindCellPoints
{
  typedef typename dax::cont::internal::FindBinding<Invocation, dax::cont::arg::Topology>::type TopoIndex;
  typedef typename dax::cont::internal::Bindings<Invocation>::template GetType<TopoIndex::value>::type TopoControlBinding;
  typedef typename TopoControlBinding::ExecArg TopoExecArgType;
  TopoExecArgType TopoExecArg;

  typedef typename dax::cont::internal::Bindings<Invocation>::template GetType<N>::type ControlBinding;
  typedef typename ControlBinding::ExecArg ExecArgType;
  typedef typename dax::cont::arg::ConceptMapTraits<ControlBinding>::Tags Tags;
  ExecArgType ExecArg;

  typedef typename ExecArgType::ValueType ComponentType;
  typedef typename TopoExecArgType::CellTag CellTag;

public:
  typedef dax::exec::CellField<ComponentType,CellTag> ValueType;
  ValueType Value;

  typedef typename boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   ValueType&,
                                   ValueType const&>::type ReturnType;
  typedef ValueType SaveType;

  DAX_CONT_EXPORT BindCellPoints(dax::cont::internal::Bindings<Invocation>& bindings):
    TopoExecArg(bindings.template Get<TopoIndex::value>().GetExecArg()),
    ExecArg(bindings.template Get<N>().GetExecArg()),
    Value() {}


  DAX_EXEC_EXPORT ReturnType operator()(
      dax::Id cellIndex,
      const dax::exec::internal::WorkletBase& work)
    {
    dax::exec::CellVertices<CellTag> pointIndices =
        this->TopoExecArg.GetPointIndices(cellIndex, work);
    for(int vertexIndex = 0;
        vertexIndex < pointIndices.NUM_VERTICES;
        ++vertexIndex)
      {
      this->Value[vertexIndex] = this->ExecArg(pointIndices[vertexIndex], work);
      }
    return this->Value;
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(
      int cellIndex,
      const dax::exec::internal::WorkletBase& worklet) const
    {
    //Look at the concept map traits. If we have the Out tag
    //we know that we must call our ExecArgs SaveExecutionResult.
    //Otherwise we are an input argument and that behavior is undefined
    //and very bad things could happen
    typedef typename Tags::
          template Has<typename dax::cont::sig::Out>::type HasOutTag;
    this->saveResult(cellIndex,worklet,HasOutTag());
    }

  //method enabled when we do have the out tag ( or InOut)
  template <typename HasOutTag>
  DAX_EXEC_EXPORT
  void saveResult(int cellIndex,
                  const dax::exec::internal::WorkletBase& work,
                  HasOutTag,
                  typename boost::enable_if<HasOutTag>::type* = 0) const
    {
    dax::exec::CellVertices<CellTag> pointIndices =
        this->TopoExecArg.GetPointIndices(cellIndex, work);
    for(int vertexIndex = 0;
        vertexIndex < pointIndices.NUM_VERTICES;
        ++vertexIndex)
      {
      this->ExecArg.SaveExecutionResult(pointIndices[vertexIndex],
                                        this->Value[vertexIndex],work);
      }
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
#endif //__dax_exec_arg_BindCellPoints_h
