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
#ifndef __dax_cont_internal_FindBinding_h
#define __dax_cont_internal_FindBinding_h
#if defined(DAX_DOXYGEN_ONLY)
#else // !defined(DAX_DOXYGEN_ONLY)

#include <dax/cont/internal/Bindings.h>
#include <dax/cont/arg/ConceptMap.h>

#include <dax/internal/Members.h>

#include <boost/mpl/if.hpp>
#include <boost/type_traits/function_traits.hpp>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/is_same.hpp>

namespace dax { namespace cont { namespace internal {

namespace detail {

template <typename Concept> struct FindBindingFailed;

template <typename Binding, typename Concept, unsigned int I, unsigned int N>
class FindBindingImpl
{
  typedef typename Binding::template GetType<I>::type ConceptMap;
  typedef typename dax::cont::arg::ConceptMapTraits<ConceptMap>::Concept BindingConcept;
public:
  typedef typename boost::mpl::if_<boost::is_same<BindingConcept, Concept>,
                                   boost::mpl::identity< boost::integral_constant<unsigned int,I> >,
                                   FindBindingImpl<Binding, Concept, I+1, N>
                                   >::type::type type;
};
template <typename Binding, typename Concept, unsigned int N>
class FindBindingImpl<Binding, Concept, N, N>
{
  typedef typename Binding::template GetType<N>::type ConceptMap;
  typedef typename dax::cont::arg::ConceptMapTraits<ConceptMap>::Concept BindingConcept;
public:
  typedef typename boost::mpl::if_<boost::is_same<BindingConcept, Concept>,
                                   boost::integral_constant<unsigned int, N>,
                                   FindBindingFailed<Concept> >::type type;
};


} // namespace detail

template <typename Invocation,
          typename Concept>
struct FindBinding
{
  enum{ len = Invocation::NUM_PARAMETERS };
  typedef typename dax::cont::internal::Bindings<Invocation>::type BindingsType;
  typedef typename detail::FindBindingImpl<BindingsType, Concept,1,len>::type type;
};



}}} // namespace dax::cont::internal

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax_cont_internal_FindBinding_h
