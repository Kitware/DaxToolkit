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
#ifndef __dax_exec_arg_BindKeyGroup_h
#define __dax_exec_arg_BindKeyGroup_h

#if defined(DAX_DOXYGEN_ONLY)
#else // !defined(DAX_DOXYGEN_ONLY)

#include <dax/cont/sig/KeyGroup.h>
#include <dax/exec/Assert.h>

namespace dax{ namespace exec { namespace arg {

template<typename IndexExecArgType,
         typename ValueExecArgType>
struct KeyGroup
{
private:
    dax::Id m_StartIndex;
    dax::Id m_Size;
    IndexExecArgType m_Indices;
    ValueExecArgType m_Values;
    dax::exec::internal::WorkletBase m_Worklet;
public:
    DAX_EXEC_EXPORT KeyGroup(
            dax::Id StartIndex,
            dax::Id Size,
            const IndexExecArgType &Indices,
            const ValueExecArgType &Values,
            const dax::exec::internal::WorkletBase& Worklet)
        : m_StartIndex(StartIndex),
          m_Size(Size),
          m_Indices(Indices),
          m_Values(Values),
          m_Worklet(Worklet){}

    DAX_EXEC_EXPORT dax::Id GetNumberOfValues() const {return m_Size;}
    DAX_EXEC_EXPORT typename ValueExecArgType::ValueType Get(dax::Id index) const
    {
        DAX_ASSERT_EXEC(index < m_Size, m_Worklet);
        dax::Id ValueIndex = m_Indices(m_StartIndex + index, m_Worklet);
        return m_Values(ValueIndex, m_Worklet);
    }
};

template <typename Invocation, int N>
class BindKeyGroup : public dax::exec::arg::ArgBase<BindKeyGroup<Invocation, N> >
{
  typedef dax::exec::arg::ArgBaseTraits< BindKeyGroup< Invocation, N > > Traits;

  typedef dax::cont::internal::Bindings<Invocation> AllControlBindings;

  typedef typename Traits::ExecArgType ExecArgType;
  typedef typename Traits::KeyCountExecArgType KeyCountExecArgType;
  typedef typename Traits::KeyOffsetExecArgType KeyOffsetExecArgType;
  typedef typename Traits::KeyIndexExecArgType KeyIndexExecArgType;

  ExecArgType ExecArg;
  KeyCountExecArgType KeyCountExecArg;
  KeyOffsetExecArgType KeyOffsetExecArg;
  KeyIndexExecArgType KeyIndexExecArg;

public:
  typedef typename Traits::ValueType ValueType;
  typedef typename Traits::ReturnType ReturnType;
  typedef typename Traits::SaveType SaveType;

  DAX_CONT_EXPORT BindKeyGroup(AllControlBindings& bindings):
    ExecArg(dax::exec::arg::GetNthExecArg<N>(bindings)),
    KeyCountExecArg(dax::exec::arg::GetNthExecArg<Traits::KeyCountIndex>(bindings)),
    KeyOffsetExecArg(dax::exec::arg::GetNthExecArg<Traits::KeyOffsetIndex>(bindings)),
    KeyIndexExecArg(dax::exec::arg::GetNthExecArg<Traits::KeyIndexIndex>(bindings))
  {}

  template<typename IndexType>
  DAX_EXEC_EXPORT
  ReturnType GetValueForReading(const IndexType& index,
                            const dax::exec::internal::WorkletBase& work) const
    {
    //Create this id's KeyGroup object and pass it back.
    return ReturnType( KeyOffsetExecArg(index, work),
                       KeyCountExecArg(index, work),
                       KeyIndexExecArg,
                       ExecArg,
                       work);
    }
};


//the traits for BindKeyGroup
template <typename Invocation,  int N >
struct ArgBaseTraits< BindKeyGroup<Invocation, N> >
{
private:
  typedef typename dax::exec::arg::BindInfo<N,Invocation> MyInfo;
  typedef typename MyInfo::Tags Tags;

  //We need the argument count to grab the last few bindings,
  //which have been special-purposed to be the parts necessary
  //to create the requested key-groups.
  enum { ArgumentCount = boost::function_traits<Invocation>::arity };

  typedef dax::exec::arg::BindInfo<ArgumentCount - 2, Invocation> KeyCountInfo;
  typedef dax::exec::arg::BindInfo<ArgumentCount - 1, Invocation> KeyOffsetInfo;
  typedef dax::exec::arg::BindInfo<ArgumentCount - 0, Invocation> KeyIndexInfo;

public:

  enum{KeyCountIndex=ArgumentCount - 2};
  enum{KeyOffsetIndex=ArgumentCount - 1};
  enum{KeyIndexIndex=ArgumentCount};


  typedef typename KeyCountInfo::ExecArgType KeyCountExecArgType;
  typedef typename KeyOffsetInfo::ExecArgType KeyOffsetExecArgType;
  typedef typename KeyIndexInfo::ExecArgType KeyIndexExecArgType;


  typedef typename MyInfo::ExecArgType ExecArgType;
  typedef typename ::boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   ::boost::true_type,
                                   ::boost::false_type>::type HasOutTag;
  typedef typename ::boost::mpl::if_<typename Tags::template Has<dax::cont::sig::In>,
                                   ::boost::true_type,
                                   ::boost::false_type>::type HasInTag;

  typedef dax::exec::arg::KeyGroup<KeyIndexExecArgType,
                                    ExecArgType> KeyGroupType;


  typedef KeyGroupType ValueType;
  typedef typename boost::mpl::if_<typename HasOutTag::type,
                                   ValueType,
                                   ValueType >::type ReturnType;
  typedef ValueType SaveType;
};


} } } //dax exec arg

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif // __dax_exec_arg_BindKeyGroup_h
