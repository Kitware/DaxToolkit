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

/// Given a worklet invocation (dax::internal::Invocation) and an argument
/// index, finds the executive arg type (the binding class located in
/// dax::exec::arg that knows how to get and set values between execution
/// environment arrays and worklet invocations) and the tags associated with
/// the control signature parameter (things typically like In, Out, Points,
/// Cells, etc.).
///
template<int Index_, typename Invocation>
struct BindInfo
{
  /// A members type that maps from control environment structure to execution
  /// environment structure.
  ///
  typedef typename dax::cont::internal::Bindings<Invocation>::type
      AllControlBindings;
private:
  typedef typename AllControlBindings::template GetType<Index_>::type
      MyControlBinding;
  typedef typename dax::cont::arg::ConceptMapTraits<MyControlBinding> MyTraits;
public:
  /// The executive arg type (the binding class located in dax::exec::arg that
  /// knows how to get and set values between execution environment arrays and
  /// worklet invocations.
  ///
  typedef typename MyControlBinding::ExecArg ExecArgType;

  /// The tags associated wtih the control signature parameter (things
  /// typically like In, Out, Points, Cells, etc.).
  ///
  typedef typename MyTraits::Tags Tags;

  enum{Index=Index_};
};

/// Given a control parameter type, find the first occurance of the execution
/// argument associated with that type. For example,
/// FindBindInfo<dax::cont::arg::Topology,Invocation>
/// will get you the information for the first occurance of
/// dax::cont::arg::Topology.
///
template<class ContType, typename Invocation>
struct FindBindInfo
{
  /// A members type that maps from control environment structure to execution
  /// environment structure.
  ///
  typedef typename dax::cont::internal::Bindings<Invocation>::type
      AllControlBindings;
private:
  typedef typename dax::cont::internal::FindBinding<Invocation,ContType>::type
      Index_;
  typedef typename AllControlBindings::template GetType<Index_::value>::type
      MyControlBinding;
  typedef typename dax::cont::arg::ConceptMapTraits<MyControlBinding> MyTraits;
public:
  /// The executive arg type (the binding class located in dax::exec::arg that
  /// knows how to get and set values between execution environment arrays and
  /// worklet invocations.
  ///
  typedef typename MyControlBinding::ExecArg ExecArgType;

  /// The tags associated wtih the control signature parameter (things
  /// typically like In, Out, Points, Cells, etc.).
  ///
  typedef typename MyTraits::Tags Tags;

  enum{Index=Index_::value};
};


template< int Index_, class BindingsType >
typename BindingsType::template GetType<Index_>::type::ExecArg //return type
GetNthExecArg(BindingsType& bindings)
{
   return bindings.template Get<Index_>().GetExecArg();
};

} } } //dax::exec::arg

#endif
