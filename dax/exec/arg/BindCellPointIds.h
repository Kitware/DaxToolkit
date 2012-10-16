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

  typedef typename boost::mpl::if_<
          typename Tags::template Has<dax::cont::sig::Out>,
                   ValueType&,
                  ValueType const&>::type ReturnType;

  DAX_CONT_EXPORT
  BindCellPointIds(dax::cont::internal::Bindings<Invocation>& bindings):
    TopoExecArg(bindings.template Get<N>().GetExecArg()),
    Value(ComponentType())
    {
    }


  DAX_EXEC_EXPORT ReturnType operator()(dax::Id id,
                          const dax::exec::internal::WorkletBase& work)
    {
    //we have to call topo to get the cell since the topology is private
    //this allows us to use a FieldMap between the bindCellPoints and
    //the real topology
    const CellType cell = this->TopoExecArg.operator()(id,work);
    this->Value = cell.GetPointIndices();
    return this->Value;
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(dax::Id id,
                          const dax::exec::internal::WorkletBase& worklet) const
    {
    this->TopoExecArg.SaveExecutionResult(id,this->Value,worklet);
    }
};

}}} // namespace dax::exec::arg

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax_exec_arg_BindCellPointIds_h
