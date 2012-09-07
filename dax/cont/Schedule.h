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
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/internal/Bindings.h>
#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/Tag.h>
#include <dax/exec/internal/Functor.h>

#include <boost/utility/enable_if.hpp>



namespace dax { namespace cont {

namespace detail
{
//forward declare schedule functors that are needed to execute worklets
//the implementation details are at the bottom of this file
template <class WorkType> class CollectCount;
template<class WorkType> class CreateExecutionResources;
}

template <class WorkletType, typename A1>
void Schedule(WorkletType w, A1 a1)
{
  //construct the signature of the worklet on the control side
  //we are stating that parameters A1 through A3 are arguments
  //that will be passed to the concept mapper to make sure we can
  //convert those control side types to the request execution side types
  typedef WorkletType ControlInvocationSignature(A1);
  typedef typename WorkletType::WorkType WorkType;


  //construct a package of all the paramters that are going to be passed to
  //to the worklet. We can match the run time type of a1, a2 to the actual
  //defintions given about the control side which now is contained inside
  //ControlInvocationSignature
  dax::cont::internal::Bindings<ControlInvocationSignature> binded(a1);

  //Activate the visitor pattern to determ the count for the number
  //of the elements we need to iterate over
  dax::Id count=0;
  binded.ForEach(dax::cont::detail::CollectCount<WorkType>(count));

  // //Activate the visitor pattern applying the CreateExecutionResources
  // //visitor to all elements in the worklet parameters class
  binded.ForEach(dax::cont::detail::CreateExecutionResources<WorkType>(count));

  dax::exec::internal::Functor<ControlInvocationSignature> func(w, binded);

  // dax::cont::internal::Schedule(f,count,DeviceAdapter());

}

template <class WorkletType, typename A1, typename A2>
void Schedule(WorkletType w, A1 a1, A2 a2)
{
  //construct the signature of the worklet on the control side
  //we are stating that parameters A1 through A3 are arguments
  //that will be passed to the concept mapper to make sure we can
  //convert those control side types to the request execution side types
  typedef WorkletType ControlInvocationSignature(A1,A2);
  typedef typename WorkletType::WorkType WorkType;


  //construct a package of all the paramters that are going to be passed to
  //to the worklet. We can match the run time type of a1, a2 to the actual
  //defintions given about the control side which now is contained inside
  //ControlInvocationSignature
  dax::cont::internal::Bindings<ControlInvocationSignature> binded(a1, a2);

  //Activate the visitor pattern to determ the count for the number
  //of the elements we need to iterate over
  dax::Id count=0;
  binded.ForEach(dax::cont::detail::CollectCount<WorkType>(count));

  // //Activate the visitor pattern applying the CreateExecutionResources
  // //visitor to all elements in the worklet parameters class
  binded.ForEach(dax::cont::detail::CreateExecutionResources<WorkType>(count));

  dax::exec::internal::Functor<ControlInvocationSignature> func(w, binded);

  // dax::cont::internal::Schedule(f,count,DeviceAdapter());

}

template <class WorkletType, typename A1, typename A2, typename A3>
void Schedule(WorkletType w, A1 a1, A2 a2, A3 a3)
{

  //construct the signature of the worklet on the control side
  //we are stating that parameters A1 through A3 are arguments
  //that will be passed to the concept mapper to make sure we can
  //convert those control side types to the request execution side types
  typedef WorkletType ControlInvocationSignature(A1,A2,A3);
  typedef typename WorkletType::WorkType WorkType;

  //construct a package of all the paramters that are going to be passed to
  //to the worklet. We can match the run time type of a1, a2 to the actual
  //defintions given about the control side which now is
  //ControlInvocationSignature
  dax::cont::internal::Bindings<ControlInvocationSignature> binded(a1, a2, a3);

  //Activate the visitor pattern to determ the count for the number
  //of the elements we need to iterate over
  dax::Id count=0;
  binded.ForEach(dax::cont::detail::CollectCount<WorkType>(count));


  // //Activate the visitor pattern applying the CreateExecutionResources
  // //visitor to all elements in the worklet parameters class
  binded.ForEach(dax::cont::detail::CreateExecutionResources<WorkType>(count));

  dax::exec::internal::Functor<ControlInvocationSignature> func(w, binded);

  // dax::cont::internal::Schedule(f,count,DeviceAdapter());
}

namespace detail
{
template <class WorkType>
class CollectCount
{
  dax::Id& Count;
public:
  CollectCount(dax::Id& c): Count(c) { this->Count = 0; }

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
