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
#include <dax/exec/CellVertices.h>
#include <dax/exec/internal/IJKIndex.h>
#include <dax/exec/internal/WorkletBase.h>

#include <boost/utility/enable_if.hpp>

namespace dax { namespace exec { namespace arg {

template <typename Invocation, int N>
class BindPermutedCellField : public dax::exec::arg::ArgBase< BindPermutedCellField<Invocation, N> >
{
  typedef dax::exec::arg::ArgBaseTraits< BindPermutedCellField< Invocation, N > > Traits;

  typedef typename Traits::TopoExecIndex TopoExecIndex;
  typedef typename Traits::TopoExecArgType TopoExecArgType;
  typedef typename Traits::ExecArgType ExecArgType;
public:

  typedef typename Traits::ValueType ValueType;
  typedef typename Traits::ReturnType ReturnType;
  typedef typename Traits::SaveType SaveType;

  DAX_CONT_EXPORT BindPermutedCellField(
      dax::cont::internal::Bindings<Invocation>& bindings):
    TopoExecArg(bindings.template Get<TopoExecIndex::value>().GetExecArg()),
    ExecArg(bindings.template Get<N>().GetExecArg()),
    Value() {}


  DAX_EXEC_EXPORT ReturnType GetValueForWriting()
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
  typedef typename dax::cont::internal::Bindings<Invocation> BindingsType;
  typedef typename dax::cont::internal::FindBinding<
      BindingsType, dax::cont::arg::Topology>::type TopoIndex;
  typedef typename BindingsType::template GetType<TopoIndex::value>::type
      TopoControlBinding;

  typedef typename dax::cont::internal::Bindings<Invocation>
      ::template GetType<N>::type ControlBinding;

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

  typedef typename ExecArgType::ValueType ValueType;
  typedef typename boost::mpl::if_<typename HasOutTag::type,
                                   ValueType&,
                                   ValueType const>::type ReturnType;
  typedef ValueType SaveType;
};

}}} // namespace dax::exec::arg

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax_exec_arg_BindPermutedCellField_h
