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

# ifndef __dax__internal__Members_h
# define __dax__internal__Members_h
# if defined(DAX_DOXYGEN_ONLY)

namespace dax { namespace internal {

/// \class Members       Members.h dax/internal/Members.h
/// \brief Store programmatically-indexable members.
///
/// \tparam TypeSequence Type sequence specifying member initializer types.
///  Currently the only supported type sequence is a function type
///  of the form \code T0(T1,T2,...)
///  \endcode where index \c 0 is the return type and greater indexes
///  are the positional parameter types.
///  If \c T0 is \c void then index \c 0 is not valid.
///
/// \tparam MemberMap    Meta-function to map index/type pairs to member types.
///  The expression \code typename MemberMap::template Get<N,TN>::type
///  \endcode must be valid for each valid index \c N
///  and corresponding \c TypeSequence element \c TN.
///  The default performs identity mapping of \c TN.
///
/// TODO: Detailed documentation.
template <typename TypeSequence, typename MemberMap = unspecified_identity_map>
class Members
{
public:
  /// \brief Initialize all members with a single value.
  /// \tparam U Type of initialization value.
  /// \param  u Initialization value.
  ///
  /// The given value is forwarded to the constructor of every member.
  /// In C++11 forwarding is perfect.
  /// In C++03 values are forwarded by reference or reference-to-const.
  template <typename U> Members(U&& u);

  /// \brief Initialize each member with its own value.
  /// \param v... Initialization values.
  ///
  /// Argument types are defined by elements of the \c TypeSequence,
  /// except that there is no argument for element 0 if it is \c void.
  /// Argument values are forwarded to the corresponding member constructors.
  /// In C++11 forwarding is perfect.
  /// In C++03 values are forwarded by \em copy.
  Members(T...v);

  /// \class GetType       Members.h dax/internal/Members.h
  /// \brief Get the type of a member.
  /// \tparam N Index of the member whose type to get.
  template <int N> class GetType
    {
    public:
    /// \brief Type of member \c N.
    ///
    /// The type is element \c N of the \c TypeSequence
    /// mapped through \c MemberMap.
    typedef type_of_Nth_member type;
    };

  /// \brief Get the value of a member by reference.
  /// \tparam N Index of the member whose value to get.
  template <int N> typename GetType<N>::type& Get();

  /// \brief Invoke a functor passing each member.
  /// \tparam Functor  Type of \c functor to invoke.
  /// \param  functor  The functor to invoke with each member.
  ///
  /// Invokes
  ///  \c functor(Get<I>())
  /// for each member \c I.
  template <typename Functor> void ForEach(Functor functor);
};

}} // namespace dax::internal

# else // !defined(DAX_DOXYGEN_ONLY)

# include <dax/internal/Configure.h>
# include <dax/internal/ExportMacros.h>

# include <dax/internal/GetNthType.h>
# include <boost/static_assert.hpp>

# if __cplusplus >= 201103L
#  include <utility>
# else // !(__cplusplus >= 201103L)
#  include <dax/internal/ParameterPackCxx03.h>
# endif // !(__cplusplus >= 201103L)

namespace dax { namespace internal {

namespace detail {

template <int Id, typename T>
class Member
{
  T M;
public:
  typedef T type;
# if __cplusplus >= 201103L
  template <typename U> Member(U&& u): M(std::forward<U>(u)) {}
# else // !(__cplusplus >= 201103L)
  template <typename U> Member(U& u): M(u) {}
  template <typename U> Member(U const& u): M(u) {}
# endif // !(__cplusplus >= 201103L)
  type& operator()() { return M; }
  type const& operator()() const { return M; }
};

struct MemberMapDefault { template <int Id, typename T> struct Get { typedef T type; }; };
} // namespace detail

template <typename Types, typename MemberMap = detail::MemberMapDefault> class Members;

namespace detail {

template <unsigned int M, unsigned int N> struct MembersForEach
{
  BOOST_STATIC_ASSERT(M<N);
  template <typename Types, typename MemberMap, typename F>
  DAX_EXEC_CONT_EXPORT static void Apply(Members<Types, MemberMap>& self, F f)
    {
    MembersForEach<M,N-1>::Apply(self, f);
    f(self.template Get<N>());
    }
};
template <unsigned int N> struct MembersForEach<N,N>
{
  template <typename Types, typename MemberMap, typename F>
  DAX_EXEC_CONT_EXPORT static void Apply(Members<Types, MemberMap>&self, F f)
    {
    f(self.template Get<N>());
    }
};

} // namespace detail

#define _dax_Members_API(I)                                             \
  template <int N>                                                      \
  class GetType                                                         \
  {                                                                     \
    typedef typename GetNthType<N, TypeSequence>::type TN;              \
    typedef typename MemberMap::template Get<N,TN>::type MTN;           \
    typedef detail::Member<N, MTN> member_type;                         \
    friend class Members;                                               \
  public:                                                               \
    typedef typename member_type::type type;                            \
  };                                                                    \
  template <int N>                                                      \
  DAX_EXEC_CONT_EXPORT typename GetType<N>::type& Get()                 \
    {                                                                   \
    typedef typename GetType<N>::member_type member_type;               \
    return static_cast<member_type&>(*this)();                          \
    }                                                                   \
  template <typename F>                                                 \
  DAX_EXEC_CONT_EXPORT void ForEach(F f)                                \
    {                                                                   \
    detail::MembersForEach<I,_dax_pp_sizeof___T>::Apply(*this, f);      \
    }

# if __cplusplus >= 201103L
#  define _dax_pp_sizeof___T sizeof...(T)

namespace detail {

template <int...> struct NumList;
template <typename,int> struct NumListNext;
template <int... Ns,int N> struct NumListNext<NumList<Ns...>,N > { typedef NumList<Ns...,N> type; };
template <int N> struct Nums { typedef typename NumListNext<typename Nums<N-1>::type,N>::type type; };
template <> struct Nums<0> { typedef NumList<> type; };

template <typename Types, typename NumList, typename MemberMap> struct MembersImpl;
template <typename T0, typename...T, int...N, typename MemberMap>
struct MembersImpl<T0(T...), NumList<N...>, MemberMap>:
  public detail::Member<0,typename MemberMap::template Get<0,T0>::type>,
  public detail::Member<N,typename MemberMap::template Get<N,T>::type>...
{
  template <typename U> MembersImpl(U&& u):
    detail::Member<0,typename MemberMap::template Get<0,T0>::type>(std::forward<U>(u)),
    detail::Member<N,typename MemberMap::template Get<N,T>::type>(std::forward<U>(u))... {}
  MembersImpl(T0&& v0, T&&...v):
    detail::Member<0,typename MemberMap::template Get<0,T0>::type>(std::forward<T0>(v0)),
    detail::Member<N,typename MemberMap::template Get<N,T>::type>(std::forward<T>(v))... {}
};
template <typename...T, int...N, typename MemberMap>
struct MembersImpl<void(T...), NumList<N...>, MemberMap>:
  public detail::Member<N,typename MemberMap::template Get<N,T>::type>...
{
  template <typename U> MembersImpl(U&& u):
    detail::Member<N,typename MemberMap::template Get<N,T>::type>(std::forward<U>(u))... {}
  MembersImpl(T&&...v):
    detail::Member<N,typename MemberMap::template Get<N,T>::type>(std::forward<T>(v))... {}
};

} // namespace detail

template <typename...T, typename MemberMap>
class Members<void(T...),MemberMap>:
  public detail::MembersImpl<void(T...), typename detail::Nums<sizeof...(T)>::type, MemberMap>
{
  typedef detail::MembersImpl<void(T...), typename detail::Nums<sizeof...(T)>::type, MemberMap> derived;
  typedef void TypeSequence(T...);
public:
  template <typename U> Members(U&& u): derived(std::forward<U>(u)) {}
  Members(T&&...v): derived(std::forward<T>(v)...) {}
  _dax_Members_API(1)
};

template <typename T0, typename...T, typename MemberMap>
class Members<T0(T...),MemberMap>: private detail::MembersImpl<T0(T...), typename detail::Nums<sizeof...(T)>::type, MemberMap>
{
  typedef detail::MembersImpl<T0(T...), typename detail::Nums<sizeof...(T)>::type, MemberMap> derived;
  typedef T0 TypeSequence(T...);
public:
  template <typename U> Members(U&& u): derived(std::forward<U>(u)) {}
  Members(T0&& v0, T&&...v):
    derived(std::forward<T0>(v0), std::forward<T>(v)...) {}
  _dax_Members_API(0)
};
#  undef _dax_pp_sizeof___T
# else // !(__cplusplus >= 201103L)
#  define _dax_Member(n) detail::Member<n, typename MemberMap::template Get<n,T___##n>::type>
#  define _dax_Member_typedef(n) typedef _dax_Member(n) Member##n;
#  define _dax_Member_init_all_u(n)  Member##n(u)
#  define _dax_Member_init_each_v(n) Member##n(v##n)
#  define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 10, <dax/internal/Members.h>))
#  include BOOST_PP_ITERATE()
#  undef _dax_Member
#  undef _dax_Member_typedef
#  undef _dax_Member_init_all_u
#  undef _dax_Member_init_each_v
# endif // !(__cplusplus >= 201103L)

# undef _dax_Members_API

}} // namespace dax::internal

# endif // !defined(DAX_DOXYGEN_ONLY)
# endif //__dax__internal__Members_h

#else // defined(BOOST_PP_IS_ITERATING)

# if _dax_pp_sizeof___T > 0
template <_dax_pp_typename___T, typename MemberMap>
class Members<void(_dax_pp_T___),MemberMap>:
  _dax_pp_enum___(_dax_Member)
{
  _dax_pp_repeat___(_dax_Member_typedef)
  typedef void TypeSequence(_dax_pp_T___);
public:
  template <typename U> Members(U& u): _dax_pp_enum___(_dax_Member_init_all_u) {}
  template <typename U> Members(U const& u): _dax_pp_enum___(_dax_Member_init_all_u) {}
  Members(_dax_pp_params___(v)): _dax_pp_enum___(_dax_Member_init_each_v) {}
  _dax_Members_API(1)
};
# endif // _dax_pp_sizeof___T > 0

template <typename T0 _dax_pp_comma _dax_pp_typename___T, typename MemberMap>
class Members<T0(_dax_pp_T___),MemberMap>:
  private detail::Member<0, typename MemberMap::template Get<0,T0>::type>
  _dax_pp_comma _dax_pp_enum___(_dax_Member)
{
  typedef detail::Member<0, typename MemberMap::template Get<0,T0>::type> Member0;
  _dax_pp_repeat___(_dax_Member_typedef)
  typedef T0 TypeSequence(_dax_pp_T___);
public:
  template <typename U> Members(U& u): Member0(u) _dax_pp_comma _dax_pp_enum___(_dax_Member_init_all_u) {}
  template <typename U> Members(U const& u): Member0(u) _dax_pp_comma _dax_pp_enum___(_dax_Member_init_all_u) {}
  Members(T0 v0 _dax_pp_comma _dax_pp_params___(v)): Member0(v0) _dax_pp_comma _dax_pp_enum___(_dax_Member_init_each_v) {}
  _dax_Members_API(0)
};

#endif // defined(BOOST_PP_IS_ITERATING)
