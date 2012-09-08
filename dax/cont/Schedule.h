#ifndef __dax_cont_Schedule_h
#define __dax_cont_Schedule_h

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

#include <dax/Types.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/internal/Bindings.h>
#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/Tag.h>

#include <boost/utility/enable_if.hpp>

namespace dax { namespace cont {

namespace detail
{
//forward declare schedule functors that are needed to execute worklets
//the implementation details are at the bottom of this file
template <class WorkType> class CollectCount;
template<class WorkType> class CreateExecutionResources;
}

// Implementation for the Schedule class for single paramter
// worklet. The rest of the implementations are handled by the preprocessor
// code that is found at the bottom of the schedule class
template <class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class Schedule
{
public:
  template <class WorkletType, typename Arg1>
  Schedule(WorkletType w, Arg1 a1)
    {
    this->operator()(w,a1);
    }

  //Note any changes to this method must be reflected in the
  //other implementation inisde Schedule.txx
  template <class WorkletType, typename Arg1>
  void operator()(WorkletType w, Arg1 a1) const
    {
    // Construct the signature of the worklet invocation on the control side.
    typedef WorkletType ControlInvocationSignature(Arg1);
    typedef typename WorkletType::WorkType WorkType;

    // Bind concrete arguments T...a to the concepts declared in the
    // worklet ControlSignature through ConceptMap specializations.
    // The concept maps also know how to make the arguments available
    // in the execution environment.
    dax::cont::internal::Bindings<ControlInvocationSignature>
      bindings(a1);

    // Visit each bound argument to determine the count to be scheduled.
    dax::Id count=1;
    bindings.ForEach(dax::cont::detail::CollectCount<WorkType>(count));

    // Visit each bound argument to set up its representation in the
    // execution environment.
    bindings.ForEach(dax::cont::detail::CreateExecutionResources<WorkType>(count));

    // Schedule the worklet invocations in the execution environment.
    dax::cont::internal::NG_Schedule<ControlInvocationSignature>
      (w, bindings, count, DeviceAdapterTag());
    }

  //this is parsed by the boost preprocessor to be the rest
  //of the implementations of the schedule constructor and operator
  #include "Schedule.txx"
};


namespace detail
{
template <class WorkType>
class CollectCount
{
  dax::Id& Count;
public:
  CollectCount(dax::Id& c): Count(c) { this->Count = 1; }

  template <typename C, typename A>
  void operator()(const dax::cont::arg::ConceptMap<C,A>& c)
    {
    //determine the concept and traits of the concept we
    //have. This will be used to look up the domains that the concept
    //has, and if one of those domains match our domina, we know we
    //can ask it for the size/len
    typedef dax::cont::arg::ConceptMap<C,A> ConceptType;
    typedef dax::cont::arg::ConceptMapTraits<ConceptType> Traits;
    typedef typename Traits::DomainTags DomainTags;
    typedef typename DomainTags::
            template Has<typename WorkType::DomainType>::type HasDomain;

    this->getCount<ConceptType,HasDomain>(c);
    }

  template <typename ConceptType, typename HasDomain>
  typename boost::enable_if<HasDomain>::type
    getCount(const ConceptType& concept)
    {
    dax::Id c = concept.GetDomainLength(typename WorkType::DomainType());
    if(c > 0)
      {
      // TODO: Consolidate counts from multiple bindings.
      // Outputs may need to be given the count to allocate.
      this->Count = c;
      }
    }

  template <typename ConceptType, typename HasDomain>
  typename boost::disable_if<HasDomain>::type
    getCount(const ConceptType&)
    {
      //this concept map doesn't have the domain tag we are interested
      //in so we ignore it
    }
};

template <class WorkType>
class CreateExecutionResources
{
protected:
  const dax::Id NumElementsToAlloc;
public:
  CreateExecutionResources(dax::Id size):
    NumElementsToAlloc(size)
    {}

  template <typename C, typename A>
  void operator()(dax::cont::arg::ConceptMap<C,A>& concept) const
    {
    //determine the concept and traits of the concept we
    //have. This will be used to look up the domains that the concept
    //has, and if one of those domains match our domina, we know we
    //can ask it for the size/len
    typedef dax::cont::arg::ConceptMap<C,A> ConceptType;
    typedef dax::cont::arg::ConceptMapTraits<ConceptType> Traits;
    typedef typename Traits::DomainTags DomainTags;
    typedef typename DomainTags::
            template Has<typename WorkType::DomainType>::type HasDomain;

    this->toExecution<ConceptType,HasDomain>(concept);
    }

  template <typename ConceptType, typename HasDomain>
  typename boost::enable_if<HasDomain>::type
    toExecution(ConceptType& concept) const
    {
    concept.ToExecution(NumElementsToAlloc);
    }

  template <typename ConceptType, typename HasDomain>
  typename boost::disable_if<HasDomain>::type
    toExecution(ConceptType&) const
    {
      //this concept map doesn't have the domain tag we are interested
      //in so we ignore it
    }
};


}


} }

#endif //__dax_cont_Schedule_h
