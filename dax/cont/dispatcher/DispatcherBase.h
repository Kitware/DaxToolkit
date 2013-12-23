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
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef __dax_cont_dispatcher_internal_DispatcherBase_h
#define __dax_cont_dispatcher_internal_DispatcherBase_h

#include <dax/Types.h>

#include <iostream>
#include <boost/type_traits/is_base_of.hpp>

#ifndef DAX_USE_VARIADIC_TEMPLATE
# include <dax/internal/ParameterPackCxx03.h>
#endif // !DAX_USE_VARIADIC_TEMPLATE

#include <dax/cont/arg/ImplementedConceptMaps.h>
#include <dax/cont/internal/Bindings.h>

#include <dax/cont/dispatcher/CollectCount.h>
#include <dax/cont/dispatcher/CreateExecutionResources.h>
#include <dax/cont/dispatcher/DetermineIndicesAndGridType.h>
#include <dax/cont/dispatcher/VerifyUserArgLength.h>

#include <dax/exec/internal/Functor.h>

namespace dax { namespace cont { namespace dispatcher {

template<typename DerivedDispatcher,
         typename WorkletBaseType_,
         typename WorkletType_,
         typename DeviceAdapterTag_>
class DispatcherBase
{
typedef typename boost::is_base_of<
        WorkletBaseType_, WorkletType_ > Worklet_Should_Match_DispatcherType;

public:
  typedef WorkletBaseType_ WorkletBaseType;
  typedef WorkletType_ WorkletType;
  typedef DeviceAdapterTag_ DeviceAdapterTag;

#ifdef DAX_USE_VARIADIC_TEMPLATE
  // Note any changes to this method must be reflected in the
  // C++03 implementation.
  template <typename...T>
  DAX_CONT_EXPORT
  void Invoke(T...arguments) const
    {
    // If you get a compile error on the following line, then you are
    // using the wrong type of Dispatcher class with your worklet.
    // (Check the type for WorkletType. It should match WorkletBaseType.)
    BOOST_MPL_ASSERT((Worklet_Should_Match_DispatcherType));

    static_cast<const DerivedDispatcher*>(this)->DoInvoke(
      this->Worklet, dax::internal::make_ParameterPack(arguments...));
    }
#else // !DAX_USE_VARIADIC_TEMPLATE
  // For C++03 use Boost.Preprocessor file iteration to simulate
  // parameter packs by enumerating implementations for all argument
  // counts.
#     define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 10, <dax/cont/dispatcher/DispatcherBase.h>))
#     include BOOST_PP_ITERATE()
#endif // !DAX_USE_VARIADIC_TEMPLATE

protected:
  DAX_CONT_EXPORT
  DispatcherBase(WorkletType worklet) : Worklet(worklet)
    { }

  template <typename WorkletType, typename ParameterPackType>
  DAX_CONT_EXPORT
  void BasicInvoke(WorkletType worklet, const ParameterPackType &arguments) const
  {
  typedef dax::cont::dispatcher::VerifyUserArgLength<WorkletType,
              ParameterPackType::NUM_PARAMETERS> WorkletUserArgs;
  //if you are getting this error you are passing less arguments than requested
  //in the control signature of this worklet
  DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::NotEnoughParameters));

  //if you are getting this error you are passing too many arguments
  //than requested in the control signature of this worklet
  DAX_ASSERT_ARG_LENGTH((typename WorkletUserArgs::TooManyParameters));

  // Bind concrete arguments T...a to the concepts declared in the
  // worklet ControlSignature through ConceptMap specializations.
  // The concept maps also know how to make the arguments available
  // in the execution environment.
  typedef dax::internal::Invocation<WorkletType,ParameterPackType> Invocation;
  typename dax::cont::internal::Bindings<Invocation>::type
      bindings = dax::cont::internal::BindingsCreate(worklet, arguments);

  // Visit each bound argument to determine the count to be scheduled.
  typedef typename WorkletType::DomainType DomainType;
  dax::Id count=1;
  bindings.ForEachCont(
        dax::cont::dispatcher::CollectCount<DomainType>(count));

  // Visit each bound argument to set up its representation in the
  // execution environment.
  bindings.ForEachCont(
        dax::cont::dispatcher::CreateExecutionResources(count));

  //Schedule the worklet invocations in the execution environment, based
  //on its type. If the worklet is a MapCell we can optimize the iteration
  //compared to a basic MapField
  dax::exec::internal::Functor<Invocation> bindingFunctor(worklet, bindings);

  typedef typename dax::cont::dispatcher::DetermineIndicesAndGridType<
                      WorkletBaseType, Invocation>  CellSchedulingIndices;

  CellSchedulingIndices cellScheduler(bindings,count);
  if(cellScheduler.isValidForGridScheduling())
    {
    // Schedule the worklet invocations in the execution environment
    // using the specialized id3 scheduler
    dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag>::
            Schedule(bindingFunctor,cellScheduler.gridCount());
    }
  else
    {
    // Schedule the worklet invocations in the execution environment.
    dax::cont::DeviceAdapterAlgorithm< DeviceAdapterTag >::
            Schedule(bindingFunctor,count);
    }
  }

private:
  WorkletType Worklet;
};

} }  } //namespace dax::cont::dispatcher_internal

#endif //__dax_cont_dispatcher_internal_DispatcherBase_h

#else // defined(BOOST_PP_IS_ITERATING)
#if _dax_pp_sizeof___T > 0
  template <_dax_pp_typename___T>
  DAX_CONT_EXPORT
  void Invoke(_dax_pp_params___(arguments)) const
    {
    // If you get a compile error on the following line, then you are
    // using the wrong type of Dispatcher class with your worklet.
    // (Check the type for WorkletType. It should match WorkletBaseType.)
    BOOST_MPL_ASSERT((Worklet_Should_Match_DispatcherType));

    static_cast<const DerivedDispatcher*>(this)->DoInvoke(
      this->Worklet,
      dax::internal::make_ParameterPack( _dax_pp_args___(arguments) ) );
    }
#     endif // _dax_pp_sizeof___T > 1
# endif // defined(BOOST_PP_IS_ITERATING)