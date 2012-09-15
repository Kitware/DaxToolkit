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
#ifndef __dax_cont_arg_ConceptMap_h
#define __dax_cont_arg_ConceptMap_h

#include <dax/cont/sig/Tag.h>
#include <dax/internal/Tags.h>

/// \namespace dax::cont::arg
/// \brief Dax Control Environment argument concepts and concept maps

namespace dax { namespace cont { namespace arg {

#if defined(DAX_DOXYGEN_ONLY)
/// \headerfile ConceptMap.h dax/cont/arg/ConceptMap.h
/// \brief Map a concrete \c Argument type to an abstract \c Concept.
template <typename Concept, typename Argument> class ConceptMap
{
public:
  /// Construct with a value convertible to the \c Argument type.
  ConceptMap(Argument value);

  /// Type representing the argument value in the execution environment.
  typedef dax::exec::arg::unspecified_type ExecArg;

  /// Get representation of argument value for the execution environment.
  ExecArg GetExecArg();
};
#else // !defined(DAX_DOXYGEN_ONLY)
template <typename Concept, typename Argument> class ConceptMap;
#endif // !defined(DAX_DOXYGEN_ONLY)

template <typename ConceptMap> class ConceptMapTraits;

/// \headerfile ConceptMap.h dax/cont/arg/ConceptMap.h
/// \brief Extract components of a ConceptMap type.
template <typename C, typename T, typename A>
class ConceptMapTraits< ConceptMap<C(T), A> >
{
public:
  /// The ConceptMap type decomposed by these traits.
  typedef ConceptMap<C(T), A> Map;

  /// Type identifying the abstract Concept.
  typedef C Concept;

  /// \brief Tags on declared parameter to which \c Argument is bound
  ///
  /// The type is an instantiation of dax::internal::Tags.
  typedef T Tags;

  /// Concrete value type bound to the \c Concept.
  typedef A Argument;

  typedef typename Map::DomainTags DomainTags;
};

template<typename DomainTag>
class SupportedDomains
{
private:
  //create a base domain tag of type sig::Tag. We can't use sig::Domain
  //as the base, as Tags::Has never returns true when checking the base type
  typedef typename dax::internal::Tags<dax::cont::sig::Domain()> TagBase;
public:
  //we append all of the concept maps tags to to the base tag type
  typedef typename TagBase::template Add<DomainTag>::type Tags;
};

template<>
class SupportedDomains<dax::cont::sig::Domain>
{
public:
  //You can't add the base to the base already
  typedef dax::internal::Tags<dax::cont::sig::Domain()>  Tags;
};


}}} // namespace dax::cont::arg

#endif //__dax_cont_arg_ConceptMap_h
