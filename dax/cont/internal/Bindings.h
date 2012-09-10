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

# if !(__cplusplus >= 201103L)
#  include <dax/internal/ParameterPackCxx03.h>
# endif // !(__cplusplus >= 201103L)

# include <dax/cont/arg/ConceptMap.h>
# include <dax/cont/sig/Tag.h>
# include <dax/internal/GetNthType.h>
# include <dax/cont/internal/Members.h>
# include <dax/internal/Tags.h>

# include <boost/mpl/identity.hpp>
# include <boost/mpl/if.hpp>
# include <boost/static_assert.hpp>
# include <boost/type_traits/function_traits.hpp>
# include <boost/type_traits/is_same.hpp>

namespace dax { namespace cont { namespace internal {

template <typename Invocation> class Bindings;

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

template <typename Invocation> class BindingsMembers;

} // namespace detail

# if __cplusplus >= 201103L

namespace detail {

template <typename R, typename... T>
struct GetConceptAndTagsImpl< R (*)(T...) >
{
  typedef R Concept;
  typedef dax::internal::Tags<sig::Tag(T...)> Tags;
};

template <typename Worklet, typename...T>
class BindingsMembers<Worklet(T...)>
{
  typedef typename Worklet::ControlSignature ControlSig;
  typedef boost::function_traits<ControlSig> ControlSigTraits;
  BOOST_STATIC_ASSERT((boost::is_same<typename ControlSigTraits::result_type, void>::value));
  BOOST_STATIC_ASSERT((ControlSigTraits::arity == sizeof...(T)));
 public:
  typedef dax::cont::internal::Members<void(T...), BindingsMemberMap<Worklet> > type;
};

} // namespace detail

template <typename Worklet, typename...T>
class Bindings<Worklet(T...)>:
  public detail::BindingsMembers<Worklet(T...)>::type
{
  typedef typename detail::BindingsMembers<Worklet(T...)>::type derived;
 public:
  Bindings(T...v): derived(std::forward<T>(v)...) {}
};
# else // !(__cplusplus >= 201103L)
#  define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 10, <dax/cont/internal/Bindings.h>))
#  include BOOST_PP_ITERATE()
# endif // !(__cplusplus >= 201103L)

}}} // namespace dax::cont::internal

# endif // !defined(DAX_DOXYGEN_ONLY)
# endif //__dax_cont_internal_Bindings_h

#else // defined(BOOST_PP_IS_ITERATING)

namespace detail {

template <typename R _dax_pp_comma _dax_pp_typename___T>
struct GetConceptAndTagsImpl< R (*)(_dax_pp_T___) >
{
  typedef R Concept;
  typedef dax::internal::Tags<sig::Tag(_dax_pp_T___)> Tags;
};

#if _dax_pp_sizeof___T > 0
template <typename Worklet, _dax_pp_typename___T>
class BindingsMembers<Worklet(_dax_pp_T___)>
{
  typedef typename Worklet::ControlSignature ControlSig;
  typedef boost::function_traits<ControlSig> ControlSigTraits;
  BOOST_STATIC_ASSERT((boost::is_same<typename ControlSigTraits::result_type, void>::value));
  BOOST_STATIC_ASSERT((ControlSigTraits::arity == _dax_pp_sizeof___T));
 public:
  typedef dax::cont::internal::Members<void(_dax_pp_T___), BindingsMemberMap<Worklet> > type;
};
#endif // _dax_pp_sizeof___T > 0

} // namespace detail

#if _dax_pp_sizeof___T > 0
template <typename Worklet, _dax_pp_typename___T>
class Bindings<Worklet(_dax_pp_T___)>:
  public detail::BindingsMembers<Worklet(_dax_pp_T___)>::type
{
  typedef typename detail::BindingsMembers<Worklet(_dax_pp_T___)>::type derived;
 public:
  Bindings(_dax_pp_params___(v)): derived(_dax_pp_args___(v)) {}
};
#endif // _dax_pp_sizeof___T > 0

#endif // defined(BOOST_PP_IS_ITERATING)
