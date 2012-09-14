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
#ifndef __dax_exec_arg_BindCellPointIds_h
#define __dax_exec_arg_BindCellPointIds_h
#if defined(DAX_DOXYGEN_ONLY)

#else // !defined(DAX_DOXYGEN_ONLY)

#include <dax/Types.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/arg/Topology.h>
#include <dax/cont/internal/Bindings.h>
#include <dax/cont/sig/Tag.h>

#include <dax/exec/internal/FieldAccess.h>
#include <dax/exec/internal/WorkletBase.h>


#include <boost/mpl/if.hpp>
#include <boost/utility/enable_if.hpp>

namespace dax { namespace exec { namespace arg {

template <typename Invocation, int N>
class BindCellPointIds
{
  typedef typename dax::cont::internal::Bindings<Invocation>::template GetType<N>::type ControlBinding;
  typedef typename ControlBinding::ExecArg TopoArgType;
  typedef typename dax::cont::arg::ConceptMapTraits<ControlBinding>::Tags Tags;
  TopoArgType TopoExecArg;

  typedef typename TopoArgType::CellType CellType;
  typedef typename dax::Id ComponentType;


public:
  typedef dax::Tuple<ComponentType,CellType::NUM_POINTS> ValueType;
  ValueType Value;

  typedef typename boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   ValueType&,
                                   ValueType const&>::type ReturnType;

  BindCellPointIds(dax::cont::internal::Bindings<Invocation>& bindings):
    TopoExecArg(bindings.template Get<N>().GetExecArg()) {}


  DAX_EXEC_EXPORT ReturnType operator()(dax::Id id,
                                        const dax::exec::internal::WorkletBase&)
    {
    const CellType cell(this->TopoExecArg.Topo,id);
    this->Value = cell.GetPointIndices();
    return this->Value;
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(dax::Id id,
                               const dax::exec::internal::WorkletBase& worklet) const
    {
    //Look at the concept map traits. If we have the Out tag
    //we know that we must call our TopoExecArgs SaveExecutionResult.
    //Otherwise we are an input argument and that behavior is undefined
    //and very bad things could happen
    typedef typename Tags::
          template Has<typename dax::cont::sig::Out>::type HasOutTag;
    this->saveResult(id,worklet,HasOutTag());
    }

  //method enabled when we do have the out tag ( or InOut)
  template <typename HasOutTag>
  DAX_EXEC_EXPORT
  void saveResult(dax::Id id,
                  dax::exec::internal::WorkletBase worklet,
                  HasOutTag,
                  typename boost::enable_if<HasOutTag>::type* = 0) const
    {
    dax::Id index = id * CellType::NUM_POINTS;
    // This only actually works if TopoExecArg is TopologyUnstructured.
    dax::exec::internal::FieldSetMultiple(this->TopoExecArg.Topo.CellConnections,
                                  index,
                                  this->Value,
                                  worklet);
    }

  template <typename HasOutTag>
  DAX_EXEC_EXPORT
  void saveResult(dax::Id,
                  dax::exec::internal::WorkletBase,
                  HasOutTag,
                  typename boost::disable_if<HasOutTag>::type* = 0) const
    {
    }
};

}}} // namespace dax::exec::arg

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax_exec_arg_BindCellPointIds_h
