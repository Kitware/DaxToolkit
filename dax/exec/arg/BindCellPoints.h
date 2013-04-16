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
#include <dax/CellTag.h>
#include <dax/VectorTraits.h>

#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/arg/Topology.h>
#include <dax/cont/internal/Bindings.h>
#include <dax/cont/internal/FindBinding.h>
#include <dax/cont/sig/Tag.h>

#include <dax/exec/arg/ArgBase.h>
#include <dax/exec/CellField.h>
#include <dax/exec/CellVertices.h>
#include <dax/exec/internal/IJKIndex.h>
#include <dax/exec/internal/WorkletBase.h>

#include <boost/utility/enable_if.hpp>

namespace dax { namespace exec { namespace arg {

template <typename Invocation, int N>
class BindCellPoints : public dax::exec::arg::ArgBase< BindCellPoints<Invocation,N> >
{
  typedef dax::exec::arg::ArgBaseTraits< BindCellPoints< Invocation, N > > Traits;

  typedef typename Traits::TopoExecIndex TopoExecIndex;
  typedef typename Traits::TopoExecArgType TopoExecArgType;
  typedef typename Traits::ExecArgType ExecArgType;
  typedef typename TopoExecArgType::CellTag CellTag;

public:

  typedef typename Traits::ValueType ValueType;
  typedef typename Traits::ReturnType ReturnType;
  typedef typename Traits::SaveType SaveType;

  DAX_CONT_EXPORT BindCellPoints(dax::cont::internal::Bindings<Invocation>& bindings):
    TopoExecArg(bindings.template Get<TopoExecIndex::value>().GetExecArg()),
    ExecArg(bindings.template Get<N>().GetExecArg()),
    Value(typename dax::VectorTraits<ValueType>::ComponentType()) {}

  DAX_EXEC_EXPORT ReturnType GetValueForWriting()
    { return this->Value; }

  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType GetValueForReading(
                                const IndexType& index,
                                const dax::exec::internal::WorkletBase& work)
    {
    const dax::exec::CellVertices<CellTag>& pointIndices =
                                            this->TopoExecArg(index, work);
    for(int vertexIndex = 0;
        vertexIndex < pointIndices.NUM_VERTICES;
        ++vertexIndex)
      {
      this->Value[vertexIndex] = this->ExecArg(pointIndices[vertexIndex],work);
      }
    return this->Value;
    }

  DAX_EXEC_EXPORT void SaveValue(int index,
                        const dax::exec::internal::WorkletBase& work) const
    {
    this->SaveValue(index,this->Value,work);
    }

  DAX_EXEC_EXPORT void SaveValue(int index, const SaveType& v,
                        const dax::exec::internal::WorkletBase& work) const
    {
    const dax::exec::CellVertices<CellTag>& pointIndices =
                                            this->TopoExecArg(index, work);
    for(int vertexIndex = 0;
        vertexIndex < pointIndices.NUM_VERTICES;
        ++vertexIndex)
      {
      this->ExecArg.SaveExecutionResult(pointIndices[vertexIndex],
                                        v[vertexIndex],
                                        work);
      }
    }
private:
  TopoExecArgType TopoExecArg;
  ExecArgType ExecArg;
  ValueType Value;
};



//the traits for BindPermutedCellField
template <typename Invocation,  int N >
struct ArgBaseTraits< BindCellPoints<Invocation, N> >
{
private:
  typedef typename dax::cont::internal::Bindings<Invocation> BindingsType;
  typedef typename dax::cont::internal::FindBinding<BindingsType, dax::cont::arg::Topology>::type TopoIndex;
  typedef typename BindingsType::template GetType<TopoIndex::value>::type TopoControlBinding;

  typedef typename dax::cont::internal::Bindings<Invocation>::template GetType<N>::type ControlBinding;
  typedef typename dax::cont::arg::ConceptMapTraits<ControlBinding>::Tags Tags;

public:
  typedef TopoIndex TopoExecIndex;
  typedef typename TopoControlBinding::ExecArg TopoExecArgType;
  typedef typename ControlBinding::ExecArg ExecArgType;

  typedef typename ::boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   ::boost::true_type,
                                   ::boost::false_type>::type HasOutTag;

  typedef typename ::boost::mpl::if_<typename Tags::template Has<dax::cont::sig::In>,
                                   ::boost::true_type,
                                   ::boost::false_type>::type HasInTag;

  typedef dax::exec::CellField<typename ExecArgType::ValueType,
                               typename TopoExecArgType::CellTag> ValueType;
  typedef typename boost::mpl::if_<typename HasOutTag::type,
                                   ValueType&,
                                   ValueType const&>::type ReturnType;
  typedef ValueType SaveType;
};

}}} // namespace dax::exec::arg

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax_exec_arg_BindCellPoints_h
