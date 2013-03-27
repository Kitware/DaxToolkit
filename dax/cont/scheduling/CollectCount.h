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
#ifndef __dax_cont_scheduling_CollectCount_h
#define __dax_cont_scheduling_CollectCount_h


#include <dax/Types.h>
#include <dax/cont/arg/ConceptMap.h>

#include <boost/utility/enable_if.hpp>

namespace dax { namespace cont { namespace scheduling {

template <class DomainType>
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
            template Has<DomainType>::type HasDomain;

    this->getCount<ConceptType,HasDomain>(c);
    }

  template <typename ConceptType, typename HasDomain>
  typename boost::enable_if<HasDomain>::type
    getCount(const ConceptType& concept)
    {
    dax::Id c = concept.GetDomainLength(DomainType());
    if(this->Count <= 1)
      {
      this->Count = c;
      }
    else if(c > 0 && c < this->Count)
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

} } }

#endif //__dax_cont_scheduling_CollectCount_h
