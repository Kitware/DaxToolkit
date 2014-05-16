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

# ifndef __dax_cont_internal_Bindings_h
# define __dax_cont_internal_Bindings_h
# if defined(DAX_DOXYGEN_ONLY)
# else // !defined(DAX_DOXYGEN_ONLY)

#include <dax/internal/Configure.h>

# ifndef DAX_USE_VARIADIC_TEMPLATE
#  include <dax/internal/ParameterPackCxx03.h>
# endif // !DAX_USE_VARIADIC_TEMPLATE

# include <dax/cont/arg/ConceptMap.h>
# include <dax/cont/sig/Tag.h>
# include <dax/internal/GetNthType.h>
# include <dax/internal/Invocation.h>
# include <dax/internal/Members.h>
# include <dax/internal/ParameterPack.h>
# include <dax/internal/Tags.h>


# include <boost/mpl/if.hpp>

namespace dax { namespace cont { namespace internal {

namespace detail {

template <typename R> struct GetConceptAndTagsImpl
{
  typedef R Concept;
  typedef dax::internal::Tags<sig::Tag()> Tags;
};

template <typename R> struct GetConceptAndTags
{
  typedef typename GetConceptAndTagsImpl<R>::Concept type
  (typename boost::mpl::if_<typename GetConceptAndTagsImpl<R>::Tags::template Has<sig::Out>,
                            boost::mpl::identity<typename GetConceptAndTagsImpl<R>::Tags>,
                            typename GetConceptAndTagsImpl<R>::Tags::template Add<sig::In> >::type::type);
};

template <class Worklet>
struct BindingsMemberMap
{
  template <int N, typename A>
  struct Get
  {
  private:
    typedef typename Worklet::ControlSignature ControlSig;
    typedef typename dax::internal::GetNthType<N, ControlSig>::type ParameterType;
    typedef typename GetConceptAndTags<ParameterType>::type ConceptAndTags;
  public:
    typedef dax::cont::arg::ConceptMap<ConceptAndTags,A> type;
  };
};

} // namespace detail

template<typename Invocation>
struct Bindings {
  typedef dax::internal::Members<
      typename Invocation::ControlInvocationSignature,
      detail::BindingsMemberMap<typename Invocation::Worklet> > type;
};

template<typename WorkletType, typename ControlInvocationParameters>
DAX_CONT_EXPORT
typename dax::cont::internal::Bindings<
    dax::internal::Invocation<WorkletType, ControlInvocationParameters> >::type
BindingsCreate(const WorkletType &daxNotUsed(worklet),
               const ControlInvocationParameters &arguments)
{
  typedef typename dax::cont::internal::Bindings<
    dax::internal::Invocation<WorkletType, ControlInvocationParameters> >::type
      BindingsType;
  return BindingsType(arguments,
                      dax::internal::MembersCopyTag(),
                      dax::internal::MembersContTag());
}

# ifdef DAX_USE_VARIADIC_TEMPLATE

namespace detail {

template <typename R, typename... T>
struct GetConceptAndTagsImpl< R (*)(T...) >
{
  typedef R Concept;
  typedef dax::internal::Tags<sig::Tag(T...)> Tags;
};

} // namespace detail

# else // !DAX_USE_VARIADIC_TEMPLATE
#  define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 10, <dax/cont/internal/Bindings.h>))
#  include BOOST_PP_ITERATE()
# endif // !DAX_USE_VARIADIC_TEMPLATE

}}} // namespace dax::cont::internal

# endif // !defined(DAX_DOXYGEN_ONLY)
# endif //__dax_cont_internal_Bindings_h

#else // defined(BOOST_PP_IS_ITERATING)

namespace detail {
using namespace dax::cont::internal::detail;

template <typename R _dax_pp_comma _dax_pp_typename___T>
struct GetConceptAndTagsImpl< R (*) (_dax_pp_T___) >
{
  typedef R Concept;
  typedef dax::internal::Tags< sig::Tag(_dax_pp_T___) > Tags;
};

} // namespace detail

#endif // defined(BOOST_PP_IS_ITERATING)
