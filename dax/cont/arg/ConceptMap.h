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
template <typename Concept, typename Argument, typename Enable=void>
class ConceptMap
{
public:
  /// Construct with a value convertible to the \c Argument type.
  ConceptMap(Argument value);

  /// Type representing the argument value in the execution environment.
  typedef dax::exec::arg::unspecified_type ExecArg;

  /// Get representation of argument value for the execution environment.
  ExecArg GetExecArg();

  /// Tag representing the domain of the data.  Should be the same as one
  /// of the domain tags in dax/cont/sig/Tag.h.
  typedef dax::cont::sig::AnyDomain DomainTag;
};
#else // !defined(DAX_DOXYGEN_ONLY)
template <typename Concept, typename Argument, typename Enable=void>
class ConceptMap;
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

  typedef typename Map::DomainTag DomainTag;
};


}}} // namespace dax::cont::arg

#endif //__dax_cont_arg_ConceptMap_h
