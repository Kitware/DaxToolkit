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
  typedef typename TopoExecArgType::CellType CellType;

public:
  typedef dax::Tuple<ComponentType,CellType::NUM_POINTS> ValueType;
  ValueType Value;

  typedef typename boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   ValueType&,
                                   ValueType const&>::type ReturnType;

  BindCellPoints(dax::cont::internal::Bindings<Invocation>& bindings):
    TopoExecArg(bindings.template Get<TopoIndex::value>().GetExecArg()),
    ExecArg(bindings.template Get<N>().GetExecArg()) {}


  DAX_EXEC_EXPORT ReturnType operator()(dax::Id id,
                        const dax::exec::internal::WorkletBase& worklet)
    {
    const CellType cell(this->TopoExecArg.Topo,id);

    const dax::Tuple<dax::Id,CellType::NUM_POINTS> ids = cell.GetPointIndices();
    for(int i=0; i < CellType::NUM_POINTS; ++i)
      {
      this->Value[i] = this->ExecArg(ids[i], worklet);
      }
    return this->Value;
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(int id,
                       const dax::exec::internal::WorkletBase& worklet) const
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
                  typename boost::enable_if<HasOutTag>::type* dummy = 0) const
    {
    (void)dummy;
    const CellType cell(this->TopoExecArg.Topo,id);
    const dax::Tuple<dax::Id,CellType::NUM_POINTS> ids = cell.GetPointIndices();
    for(int i=0; i < CellType::NUM_POINTS; ++i)
      {
      this->ExecArg.SaveExecutionResult(ids[i],this->Value[i],worklet);
      }
    }

  template <typename HasOutTag>
  DAX_EXEC_EXPORT
  void saveResult(int,
                  const dax::exec::internal::WorkletBase&,
                  HasOutTag,
                  typename boost::disable_if<HasOutTag>::type* dummy = 0) const
    {
    (void)dummy;
    }
};

}}} // namespace dax::exec::arg

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax_exec_arg_BindCellPoints_h
