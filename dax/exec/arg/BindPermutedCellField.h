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

#include <dax/exec/arg/ArgBase.h>
#include <dax/exec/arg/BindInfo.h>
#include <dax/exec/CellVertices.h>
#include <dax/exec/internal/IJKIndex.h>
#include <dax/exec/internal/WorkletBase.h>

#include <boost/utility/enable_if.hpp>

namespace dax { namespace exec { namespace arg {

template <typename Invocation, int N>
class BindPermutedCellField : public dax::exec::arg::ArgBase< BindPermutedCellField<Invocation, N> >
{
  typedef dax::exec::arg::ArgBaseTraits< BindPermutedCellField< Invocation, N > > Traits;

  enum{TopoIndex=Traits::TopoIndex};
  typedef typename Traits::TopoExecArgType TopoExecArgType;
  typedef typename Traits::ExecArgType ExecArgType;
public:

  typedef typename Traits::ValueType ValueType;
  typedef typename Traits::ReturnType ReturnType;
  typedef typename Traits::SaveType SaveType;

  DAX_CONT_EXPORT BindPermutedCellField(
      dax::cont::internal::Bindings<Invocation>& bindings):
    TopoExecArg(dax::exec::arg::GetNthExecArg<TopoIndex>(bindings)),
    ExecArg(dax::exec::arg::GetNthExecArg<N>(bindings)),
    Value() {}


  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType GetValueForWriting(const IndexType&,
                            const dax::exec::internal::WorkletBase&)
    { return this->Value; }


  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType GetValueForReading(
                            const IndexType& index,
                            const dax::exec::internal::WorkletBase& work) const
    {
    const dax::Id cellIndex = this->TopoExecArg.GetMapIndex(index, work);
    return this->ExecArg(cellIndex, work);
    }

  DAX_EXEC_EXPORT void SaveValue(int index,
                        const dax::exec::internal::WorkletBase& work) const
    {
    this->SaveValue(index,this->Value,work);
    }

  DAX_EXEC_EXPORT void SaveValue(int index, const SaveType& v,
                        const dax::exec::internal::WorkletBase& work) const
    {
    const dax::Id cellIndex = this->TopoExecArg.GetMapIndex(index, work);
    this->ExecArg.SaveExecutionResult(cellIndex, v, work);
    }
private:
  TopoExecArgType TopoExecArg;
  ExecArgType ExecArg;
  ValueType Value;
};

//the traits for BindPermutedCellField
template <typename Invocation,  int N >
struct ArgBaseTraits< BindPermutedCellField<Invocation, N> >
{
private:
  typedef typename dax::exec::arg::FindBindInfo<dax::cont::arg::Topology,
                                               Invocation> TopoInfo;
  typedef typename dax::exec::arg::BindInfo<N,Invocation> MyInfo;
  typedef typename MyInfo::Tags Tags;

public:
  enum{TopoIndex=TopoInfo::Index};

  typedef typename TopoInfo::ExecArgType TopoExecArgType;
  typedef typename MyInfo::ExecArgType ExecArgType;

  typedef typename ::boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   ::boost::true_type,
                                   ::boost::false_type>::type HasOutTag;

  typedef typename ::boost::mpl::if_<typename Tags::template Has<dax::cont::sig::In>,
                                   ::boost::true_type,
                                   ::boost::false_type>::type HasInTag;

  typedef typename ExecArgType::ValueType ValueType;
  typedef typename boost::mpl::if_<typename HasOutTag::type,
                                   ValueType&,
                                   ValueType const>::type ReturnType;
  typedef ValueType SaveType;
};

}}} // namespace dax::exec::arg

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax_exec_arg_BindPermutedCellField_h
