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

# ifndef __dax__internal__Tags_h
# define __dax__internal__Tags_h
# if defined(DAX_DOXYGEN_ONLY)

namespace dax { namespace internal {

/// \class Tags          Tags.h dax/internal/Tags.h
/// \tparam TypeSequence Type sequence from which to extract elements.
///  Currently the only supported type sequence is a function type
///  of the form \code TagBase(Tag1,Tag2,...)
///  \endcode where the return type is element \c 0 in the sequence.
///
/// \brief Compile-time collection and testing of a tag hierarchy.
///
/// Element \c 0 in \c TypeSequence specifies a base class type
/// identifying a group of possible tags.
/// Additional elements in the sequence specify tag class types
/// driving from the base class type.
template <typename TypeSequence>
class Tags
{
public:
  /// \class Has          Tags.h dax/internal/Tags.h
  /// \tparam Tag Tag type to test.
  ///
  /// Test for presence of a given \c Tag.
  /// Derives from boost::true_type or boost::false_type depending
  /// on whether the test \c Tag is in the collection of Tags.
  template <typename Tag>
  struct Has: public boost_truth_type
  {
  };

  /// \class Add          Tags.h dax/internal/Tags.h
  /// \tparam TagOrTags   \c Tag type, type sequence, or \c Tags<> to add
  ///
  /// Compute a new \c Tags<> type that incorporates additional tags.
  template <typename TagOrTags>
  struct Add
  {
    /// New \c Tags<> type that incorporates specified \c TagOrTags.
    typedef new_Tags_type type;
  };
};

}} // namespace dax::internal

# else // !defined(DAX_DOXYGEN_ONLY)

# include <dax/internal/Configure.h>

# ifndef DAX_USE_VARIADIC_TEMPLATE
#  include <dax/internal/ParameterPackCxx03.h>
# endif // !DAX_USE_VARIADIC_TEMPLATE

# include <boost/mpl/identity.hpp>
# include <boost/mpl/if.hpp>
# include <boost/mpl/or.hpp>
# include <boost/static_assert.hpp>
# include <boost/type_traits/is_base_and_derived.hpp>

namespace dax { namespace internal {

template <typename T> class Tags;

namespace detail {

template <typename T, bool> struct TagsCheckImpl {};
template <typename T> struct TagsCheckImpl<T, true> { typedef T type; };
template <typename B, typename T> struct TagsCheck: public TagsCheckImpl<T, boost::is_base_and_derived<B,T>::value>
  { typedef T type; };

template <typename T> struct TagsBase;
template <typename B> struct TagsBase<B()> { typedef B base_type; };

template <typename Tag, typename Tags> struct TagsAddImpl;
template <typename Tags1, typename TagOrTags> class TagsAdd;

template <typename Tags1, typename Tag> class TagsAdd<Tags<Tags1>, Tag>
{
  typedef typename Tags<Tags1>::base_type base_type;
  typedef typename TagsCheck<base_type,Tag>::type tag_type;
public:
  typedef Tags<typename boost::mpl::if_<typename Tags<Tags1>::template Has<Tag>,
                                        boost::mpl::identity< Tags1 >,
                                        TagsAddImpl<Tag,Tags1>
                                        >::type::type> type;
};

template <typename Tags1, typename B> class TagsAdd< Tags<Tags1>, B()>
{
  typedef typename Tags<Tags1>::base_type base_type;
#ifndef _WIN32
  BOOST_STATIC_ASSERT((boost::mpl::or_<boost::is_same<base_type, B>,
                                       boost::is_base_and_derived<base_type, B> >::value));
#endif
public:
  typedef Tags<Tags1> type;
};

# ifdef DAX_USE_VARIADIC_TEMPLATE
template <typename B, typename T, typename...Ts> struct TagsBase<B(T,Ts...)>: TagsCheck<B,T>::type, TagsCheck<B,Ts>::type... { typedef B base_type; };
template <typename Tag, typename B, typename...Ts> struct TagsAddImpl<Tag, B(Ts...)> { typedef B type(Ts...,Tag); };
template <typename Tags1, typename B, typename T, typename...Ts> class TagsAdd< Tags<Tags1>, B(T,Ts...)>: public TagsAdd<typename TagsAdd<Tags<Tags1>,T>::type, B(Ts...)> {};
# else // !DAX_USE_VARIADIC_TEMPLATE
#  define _dax_TagsCheck(n) TagsCheck<B,T___##n>::type
#  define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 10, <dax/internal/Tags.h>))
#  include BOOST_PP_ITERATE()
#  undef _dax_TagsCheck
# endif // !DAX_USE_VARIADIC_TEMPLATE

template <typename Tags1, typename Tags2> class TagsAdd< Tags<Tags1>, Tags<Tags2> >: public TagsAdd<Tags<Tags1>,Tags2> {};

template <typename Tags, typename Tag> struct TagsHas;
template <typename T, typename Tag> struct TagsHas<Tags<T>,Tag>: public boost::is_base_and_derived<Tag, Tags<T> > {};

} // namespace detail

template <typename T> class Tags: public detail::TagsBase<T>
{
public:
  template <typename Tag> struct Has: public detail::TagsHas<Tags<T>, Tag> {};
  template <typename TagOrTags> struct Add: public detail::TagsAdd<Tags<T>, TagOrTags> {};
};

}} // namespace dax::internal

# endif // !defined(DAX_DOXYGEN_ONLY)
# endif //__dax__internal__Tags_h

#else // defined(BOOST_PP_IS_ITERATING)

template <typename Tag, typename B _dax_pp_comma _dax_pp_typename___T> struct TagsAddImpl<Tag, B(_dax_pp_T___)> { typedef B type(_dax_pp_T___ _dax_pp_comma Tag); };
template <typename Tags1, typename B, typename T1 _dax_pp_comma _dax_pp_typename___T> class TagsAdd< Tags<Tags1>, B(T1 _dax_pp_comma _dax_pp_T___)>: public TagsAdd<typename TagsAdd<Tags<Tags1>,T1>::type, B(_dax_pp_T___)> {};
#if _dax_pp_sizeof___T > 0
template <typename B, _dax_pp_typename___T> struct TagsBase<B(_dax_pp_T___)>: _dax_pp_enum___(_dax_TagsCheck) { typedef B base_type; };
#endif // _dax_pp_sizeof___T > 0

#endif // defined(BOOST_PP_IS_ITERATING)
