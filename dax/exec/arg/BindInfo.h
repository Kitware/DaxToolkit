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


#ifndef __dax_exec_arg_BindInfo_h
#define __dax_exec_arg_BindInfo_h

#include <dax/cont/internal/Bindings.h>
#include <dax/cont/internal/FindBinding.h>
#include <dax/cont/sig/Tag.h>


namespace dax { namespace exec { namespace arg {

template<int Index_, class Invocation>
struct BindInfo
{
  typedef dax::cont::internal::Bindings<Invocation> AllControlBindings;
private:
  typedef typename AllControlBindings::template GetType<Index_>::type MyControlBinding;
  typedef typename dax::cont::arg::ConceptMapTraits<MyControlBinding> MyTraits;
public:
  typedef typename MyControlBinding::ExecArg ExecArgType;
  typedef typename MyTraits::Tags Tags;
  enum{Index=Index_};
};

//Set ContType to the ControlSignature parameter type that you are
//searching for. After that you can extract the exec::arg:: object type
//for example FindBindInfo<dax::cont::arg::Topology,Sig> will get you
//the info for the first occurance of dax::cont::arg::Topology
template< class ContType, class Invocation>
struct FindBindInfo
{
  typedef dax::cont::internal::Bindings<Invocation> AllControlBindings;
private:
  typedef typename dax::cont::internal::FindBinding<AllControlBindings,ContType>::type Index_;
  typedef typename AllControlBindings::template GetType<Index_::value>::type MyControlBinding;
  typedef typename dax::cont::arg::ConceptMapTraits<MyControlBinding> MyTraits;
public:
  typedef typename MyControlBinding::ExecArg ExecArgType;
  typedef typename MyTraits::Tags Tags;
  enum{Index=Index_::value};
};


template< int Index_, class BindingsType >
typename BindingsType::template GetType<Index_>::type::ExecArg //return type
GetNthExecArg(BindingsType& bindings)
{
   return bindings.template Get<Index_>().GetNthExecArg();
};

} } } //dax::exec::arg

#endif