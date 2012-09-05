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

namespace dax { namespace exec { namespace arg {

template <typename Invocation, int N>
struct BindCellPoints
{
  typedef typename dax::cont::internal::FindBinding<Invocation, dax::cont::arg::Topology>::type TopoIndex;
  typedef typename dax::cont::internal::Bindings<Invocation>::template GetType<TopoIndex::value>::type TopoControlBinding;
  typedef typename TopoControlBinding::ExecArg TopoExecArgType;
  TopoExecArgType TopoExecArg;

  typedef typename dax::cont::internal::Bindings<Invocation>::template GetType<N>::type ControlBinding;
  typedef typename ControlBinding::ExecArg ExecArgType;
  typedef typename dax::cont::arg::ConceptMapTraits<ControlBinding>::Tags Tags;
  ExecArgType ExecArg;

  BindCellPoints(dax::cont::internal::Bindings<Invocation>& bindings):
    TopoExecArg(bindings.template Get<TopoIndex::value>()),
    ExecArg(bindings.template Get<N>()) {}

  typedef typename ExecArgType::ValueType ComponentType;

  typedef typename TopoControlBinding::CellPointsContainer CellPointsContainer;
  typedef dax::Tuple<ComponentType,CellPointsContainer::NUM_COMPONENTS> ValueType;

  typedef typename boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   ValueType&,
                                   ValueType const&>::type ReferenceType;

  ValueType Value;

  DAX_EXEC_CONT_EXPORT ReferenceType operator()(dax::Id id)
    {
    CellPointsContainer ids = this->TopoExecArg.GetCellPoints(id);
    for(int i=0; i < CellPointsContainer::NUM_COMPONENTS; ++i)
      {
      this->Value[i] = this->ExecArg(ids[i]);
      }
    return this->Value;
    }
};

}}} // namespace dax::exec::arg

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax_exec_arg_BindCellPoints_h
