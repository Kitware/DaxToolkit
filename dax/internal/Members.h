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
#ifndef __dax__internal__Members_h
#define __dax__internal__Members_h

#include <dax/internal/ParameterPack.h>

#include <dax/internal/GetNthType.h>

#include <boost/mpl/if.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/utility/enable_if.hpp>

namespace dax {
namespace internal {

namespace detail {

struct MemberMapDefault {
  template <int Id, typename T> struct Get { typedef T type; };
};

} // namespace detail (in dax::internal)

/// A tag structure used to represent a member with a null type (typically a
/// void return). This is used in place of void to prevent compile problems.
///
struct NullMember {
  DAX_EXEC_CONT_EXPORT
  NullMember() {  }

  /// This constructor simply ignores whatever argument it gives it. Once
  /// again, this is for convienience so that when NullMember is used as a
  /// template argument, the compile does not fail when you try to construct
  /// it.
  ///
  template<typename T>
  DAX_EXEC_CONT_EXPORT
  NullMember(const T &) {  }
};

/// A tag structure used in Members constructors to declare that the argument
/// contains another Members or a ParameterPack structure with arguments that
/// should be copied (explicitly) into the constructed Members class.
///
struct MembersCopyTag {  };

/// A tag structure used in Members constructors to declare that the argument
/// should be passed to all arguments and the return value (if non null) as an
/// initializer. This will allow arguments in one Members object to reference
/// all arguments of another Members object.
///
struct MembersInitialArgumentTag {  };


/// Used to overload a \c Members method that is exported only in the control
/// environment.
///
struct MembersContTag {  };

/// Used to overload a \c Members method that is exported in both the control
/// and exection environments.
///
struct MembersExecContTag {  };

template<typename Signature, typename MemberMap = detail::MemberMapDefault>
class Members;

namespace detail {

template<typename MemberMap, typename RealReturnType>
struct MembersReturnType
{
  typedef typename MemberMap::template Get<0,RealReturnType>::type type;
};
template<typename MemberMap>
struct MembersReturnType<MemberMap, void>
{
  typedef NullMember type;
};
template<typename MemberMap>
struct MembersReturnType<MemberMap, NullMember>
{
  typedef NullMember type;
};

template<typename InputParameterPack,
         typename ConstructedParameterPack,
         typename MemberMap,
         int Index>
struct MembersParameterMapConstruct
{
  typedef typename MembersParameterMapConstruct<
      InputParameterPack,
      typename dax::internal::detail::PPackPrepend<
          typename MemberMap::template Get<
              Index,
              typename InputParameterPack::template Parameter<Index>::type>::type,
          ConstructedParameterPack>::type,
      MemberMap,
      Index-1>::type type;
};

template<typename InputParameterPack,
         typename ConstructedParameterPack,
         typename MemberMap>
struct MembersParameterMapConstruct<
    InputParameterPack,
    ConstructedParameterPack,
    MemberMap,
    0>
{
  typedef ConstructedParameterPack type;
};

template<typename InputSignature, typename MemberMap>
struct MembersParameterMap
{
private:
  typedef typename dax::internal::SignatureToParameterPack<InputSignature>::type
      InputParameterPack;
public:
  typedef typename MembersParameterMapConstruct<
      InputParameterPack,
      dax::internal::ParameterPack<>,
      MemberMap,
      InputParameterPack::NUM_PARAMETERS>::type type;
};

template<typename ReturnType>
struct MembersFirstIndex
{
  static const int INDEX = 0;
};
template<>
struct MembersFirstIndex<NullMember>
{
  static const int INDEX = 1;
};

template<int N, typename MembersType>
struct MembersGetType
{
  BOOST_STATIC_ASSERT(N >= 0);
  BOOST_STATIC_ASSERT(N < MembersType::NUM_MEMBERS);
  typedef
    typename MembersType::ParameterPackType::template Parameter<N>::type type;
};
template<typename MembersType>
struct MembersGetType<0,MembersType>
{
public:
  typedef typename MembersType::ReturnType type;
};

template <unsigned int M, unsigned int N>
struct MembersForEach
{
  BOOST_STATIC_ASSERT(M<N);
  template <typename Types, typename MemberMap, typename F>
  DAX_CONT_EXPORT static void ApplyCont(Members<Types, MemberMap>& self, F f)
    {
    MembersForEach<M,N-1>::ApplyCont(self, f);
    f(self.template Get<N>());
    }
  template <typename Types, typename MemberMap, typename F>
  DAX_EXEC_EXPORT static void ApplyExec(Members<Types, MemberMap>& self, F f)
    {
    MembersForEach<M,N-1>::ApplyExec(self, f);
    f(self.template Get<N>());
    }
};
template <unsigned int N>
struct MembersForEach<N,N>
{
  template <typename Types, typename MemberMap, typename F>
  DAX_CONT_EXPORT static void ApplyCont(Members<Types, MemberMap>&self, F f)
    {
    f(self.template Get<N>());
    }
  template <typename Types, typename MemberMap, typename F>
  DAX_EXEC_EXPORT static void ApplyExec(Members<Types, MemberMap>&self, F f)
    {
    f(self.template Get<N>());
    }
};

struct MembersReturnTag {  };
struct MembersParameterTag {  };

template<int N>
struct MembersFindPositionTag {
  typedef MembersParameterTag type;
};
template<>
struct MembersFindPositionTag<0> {
  typedef MembersReturnTag type;
};

template<int N, typename Signature, typename MemberMap>
struct MembersGet
{
private:
  typedef Members<Signature,MemberMap> MembersType;
  typedef typename MembersType::template GetType<N>::type BasicType;
  typedef typename boost::remove_reference<BasicType>::type BaseType;

public:
  DAX_EXEC_CONT_EXPORT
  BaseType &
  operator()(MembersType &members, MembersParameterTag) const
  {
    BOOST_STATIC_ASSERT(N > 0);
    BOOST_STATIC_ASSERT(N < MembersType::NUM_MEMBERS);
    return members.GetArgumentValues().template GetArgument<N>();
  }

  DAX_EXEC_CONT_EXPORT
  const BaseType &
  operator()(const MembersType &members, MembersParameterTag) const
  {
    BOOST_STATIC_ASSERT(N > 0);
    BOOST_STATIC_ASSERT(N < MembersType::NUM_MEMBERS);
    return members.GetArgumentValues().template GetArgument<N>();
  }

  DAX_EXEC_CONT_EXPORT
  BaseType &
  operator()(MembersType &members, MembersReturnTag) const
  {
    BOOST_STATIC_ASSERT(N == 0);
    return members.GetReturnValue();
  }

  DAX_EXEC_CONT_EXPORT
  const BaseType &
  operator()(const MembersType &members, MembersReturnTag) const
  {
    BOOST_STATIC_ASSERT(N == 0);
    return members.GetReturnValue();
  }
};

template<typename T>
struct MembersSet
{
  T Value;
  MembersSet(T value) : Value(value) {  }
  template<typename U>
  void operator()(U &member) const {
    member = this->Value;
  }
};

} // namespace detail (in dax::internal)

/// \class Members       Members.h dax/internal/Members.h
/// \brief Store programmatically-indexable members.
///
/// The intention of the Members class is to enable the transformation from one
/// function call to another. For example, taking arguments passed to a control
/// environment function and converting them to arguments that can be passed to
/// an execution environment function (and back). The transformation of input
/// arguments (specified by the \c Signature template parameter) is determined
/// with the \c MemberMap template parameter. The runtime transformation can be
/// invoked with the \c ForEachCont or \c ForEachExec methods.
///
/// \tparam Signature Type sequence specifying member initializer types as
/// specified in a function type signature of the form \code T0(T1,T2,...)
/// \endcode where index \c 0 is the return type and greater indexes are the
/// positional parameter types. If \c T0 is \c void then index \c 0 is not
/// valid.
///
/// \tparam MemberMap Meta-function to map index/type pairs to member types.
/// The expression \code typename MemberMap::template Get<N,TN>::type \endcode
/// must be valid for each valid index \c N and corresponding \c TypeSequence
/// element \c TN. The default performs identity mapping of \c TN.
///
template<typename Signature, typename MemberMap>
class Members
{
  typedef Members<Signature, MemberMap> ThisType;
public:

  /// \brief Return type member (after mapping).
  /// Set to the TypeSequence return type after being mapped through MemberMap.
  /// If the return type is null, ReturnType is set to the type NullMember
  typedef typename detail::MembersReturnType<
      MemberMap,typename dax::internal::GetNthType<0,Signature>::type>::type
    ReturnType;

  /// \brief ParameterPack type used for holding members.
  /// This ParameterPack contains all the types of TypeSequence after being
  /// mapped through MemberMap.
  typedef typename detail::MembersParameterMap<Signature,MemberMap>::type
      ParameterPackType;

  DAX_EXEC_CONT_EXPORT
  Members() {  }

  DAX_EXEC_CONT_EXPORT
  Members(const ThisType &src)
    : ReturnValue(src.ReturnValue), ArgumentValues(src.ArgumentValues) {  }

  DAX_EXEC_CONT_EXPORT
  ThisType &operator=(const ThisType &src) {
    this->ArgumentValues = src.ArgumentValues;
    this->ReturnValue = src.ReturnValue;
    return *this;
  }

  /// Initializes the argument members with those given in a parameter pack,
  /// copying one by one.
  ///
  template<typename ParameterPackCopyType>
  DAX_EXEC_CONT_EXPORT
  Members(const ParameterPackCopyType &arguments,
          MembersCopyTag,
          MembersExecContTag)
    : ArgumentValues(arguments,
                     ParameterPackCopyTag(),
                     ParameterPackExecContTag())
  {  }
  template<typename ParameterPackCopyType>
  DAX_CONT_EXPORT
  Members(const ParameterPackCopyType &arguments,
          MembersCopyTag,
          MembersContTag)
    : ArgumentValues(arguments,
                     ParameterPackCopyTag(),
                     ParameterPackContTag())
  {  }

  /// Initializes the argument members with the given return value and those
  /// given in a parameter pack, copying one by one.
  ///
  template<typename ParameterPackCopyType>
  DAX_EXEC_CONT_EXPORT
  Members(ReturnType returnValue,
          const ParameterPackCopyType &arguments,
          MembersCopyTag,
          MembersExecContTag)
    : ReturnValue(returnValue),
      ArgumentValues(arguments,
                     ParameterPackCopyTag(),
                     ParameterPackExecContTag())
  {  }
  template<typename ParameterPackCopyType>
  DAX_CONT_EXPORT
  Members(ReturnType returnValue,
          const ParameterPackCopyType &arguments,
          MembersCopyTag,
          MembersContTag)
    : ReturnValue(returnValue),
      ArgumentValues(arguments,
                     ParameterPackCopyTag(),
                     ParameterPackContTag())
  {  }

  /// \brief Initializes this class with members in a class.
  /// The source class may have a different signature or map, but the types of
  /// all members must match up.
  template<typename CopySig, typename CopyMap>
  DAX_EXEC_CONT_EXPORT
  Members(const Members<CopySig,CopyMap> &src,
          MembersCopyTag,
          MembersExecContTag)
    : ReturnValue(src.template Get<0>()),
      ArgumentValues(src.GetArgumentValues(),
                     ParameterPackCopyTag(),
                     ParameterPackExecContTag())
  {  }
  template<typename CopySig, typename CopyMap>
  DAX_CONT_EXPORT
  Members(const Members<CopySig,CopyMap> &src,
          MembersCopyTag,
          MembersContTag)
    : ReturnValue(src.template Get<0>()),
      ArgumentValues(src.GetArgumentValues(),
                     ParameterPackCopyTag(),
                     ParameterPackContTag())
  {  }

  /// \breif Initalizes all members with the given object.
  /// The given object is passed to the constructor for the return value (if
  /// non-null) and all arguments constructed.
  ///
  template<typename InitialMember>
  DAX_EXEC_CONT_EXPORT
  Members(InitialMember &initial,
          MembersInitialArgumentTag,
          MembersExecContTag)
    : ReturnValue(initial),
      ArgumentValues(initial,
                     ParameterPackInitialArgumentTag(),
                     ParameterPackExecContTag())
  {  }
  template<typename InitialMember>
  DAX_CONT_EXPORT
  Members(InitialMember &initial,
          MembersInitialArgumentTag,
          MembersContTag)
    : ReturnValue(initial),
      ArgumentValues(initial,
                     ParameterPackInitialArgumentTag(),
                     ParameterPackContTag())
  {  }
  template<typename InitialMember>
  DAX_EXEC_CONT_EXPORT
  Members(const InitialMember &initial,
          MembersInitialArgumentTag,
          MembersExecContTag)
    : ReturnValue(initial),
      ArgumentValues(initial,
                     ParameterPackInitialArgumentTag(),
                     ParameterPackExecContTag())
  {  }
  template<typename InitialMember>
  DAX_CONT_EXPORT
  Members(const InitialMember &initial,
          MembersInitialArgumentTag,
          MembersContTag)
    : ReturnValue(initial),
      ArgumentValues(initial,
                     ParameterPackInitialArgumentTag(),
                     ParameterPackContTag())
  {  }

  /// \brief Number of members in the Signature (template parameter).
  /// This number is specifically one more than the largest index that can be
  /// provided to GetType or Get (regardless of the value of FIRST_INDEX).
  ///
  static const int NUM_MEMBERS = ParameterPackType::NUM_PARAMETERS + 1;

  /// \brief The first valid index for a member.
  ///
  /// Assuming that the return type of the Signature (and, consequently, the
  /// ReturnType typedef) is a valid type, \c FIRST_INDEX is set to 0 as the
  /// first valid index. However, if this return type is void, then \c Get<0>
  /// is not valid, so \c FIRST_INDEX is set to 1. Thus, to iterate over all
  /// possible members, consider indices [FIRST_INDEX,NUM_MEMBERS).
  ///
  static const int FIRST_INDEX = detail::MembersFirstIndex<ReturnType>::INDEX;

  /// \class GetType       Members.h dax/internal/Members.h
  /// \brief Get the type of a member.
  /// \tparam N Index of the member whose type to get.  Index 0 is the
  ///           return type whereas 1.. are the arguments.
  template<int N>
  struct GetType
  {
    BOOST_STATIC_ASSERT(N >= 0);
    BOOST_STATIC_ASSERT(N < NUM_MEMBERS);
    typedef typename detail::MembersGetType<N,ThisType>::type type;
  };

  /// \brief Get the value of a member by reference.
  /// \tparam N Index of the member whose value to get.
  template<int N>
  DAX_EXEC_CONT_EXPORT
  typename boost::remove_reference<typename GetType<N>::type>::type &
  Get()
  {
    BOOST_STATIC_ASSERT(N >= 0);
    BOOST_STATIC_ASSERT(N < NUM_MEMBERS);
    return detail::MembersGet<N,Signature,MemberMap>()(
          *this, typename detail::MembersFindPositionTag<N>::type());
  }
  template<int N>
  DAX_EXEC_CONT_EXPORT
  const typename boost::remove_reference<typename GetType<N>::type>::type &
  Get() const
  {
    BOOST_STATIC_ASSERT(N >= 0);
    BOOST_STATIC_ASSERT(N < NUM_MEMBERS);
    return detail::MembersGet<N,Signature,MemberMap>()(
          *this, typename detail::MembersFindPositionTag<N>::type());
  }

  DAX_EXEC_CONT_EXPORT
  typename boost::remove_reference<ReturnType>::type &
  GetReturnValue()
  {
    return this->ReturnValue;
  }
  DAX_EXEC_CONT_EXPORT
  const typename boost::remove_reference<ReturnType>::type &
  GetReturnValue() const
  {
    return this->ReturnValue;
  }

  DAX_EXEC_CONT_EXPORT
  ParameterPackType &GetArgumentValues()
  {
    return this->ArgumentValues;
  }
  DAX_EXEC_CONT_EXPORT
  const ParameterPackType &GetArgumentValues() const
  {
    return this->ArgumentValues;
  }

  /// \brief Invoke a functor passing each member.
  /// \tparam Functor  Type of \c functor to invoke.
  /// \param  functor  The functor to invoke with each member.
  ///
  /// Invokes
  ///  \c functor(Get<I>())
  /// for each member \c I.
  template <typename Functor>
  DAX_CONT_EXPORT
  void ForEachCont(Functor functor) {
    detail::MembersForEach<FIRST_INDEX,NUM_MEMBERS-1>::
        ApplyCont(*this, functor);
  }
  template <typename Functor>
  DAX_EXEC_EXPORT
  void ForEachExec(Functor functor) {
    detail::MembersForEach<FIRST_INDEX,NUM_MEMBERS-1>::
        ApplyExec(*this, functor);
  }

  template<typename CopySig, typename CopyMap>
  DAX_EXEC_CONT_EXPORT
  void Copy(const Members<CopySig,CopyMap> &src)
  {
    this->CopyImpl<NUM_MEMBERS-1>(src);
  }

private:
  ReturnType ReturnValue;
  ParameterPackType ArgumentValues;

  template<int N, typename CopySig, typename CopyMap>
  DAX_EXEC_CONT_EXPORT
  typename boost::disable_if_c<N == FIRST_INDEX-1>::type
  CopyImpl(const Members<CopySig,CopyMap> &src)
  {
    typedef Members<CopySig,CopyMap> SrcType;
    BOOST_STATIC_ASSERT(FIRST_INDEX == SrcType::FIRST_INDEX);
    BOOST_STATIC_ASSERT(NUM_MEMBERS == SrcType::NUM_MEMBERS);
    BOOST_STATIC_ASSERT(N >= FIRST_INDEX);
    BOOST_STATIC_ASSERT(N < NUM_MEMBERS);
    this->CopyImpl<N-1>(src);
    this->Get<N>() = src.template Get<N>();
  }

  template<int N, typename CopySig, typename CopyMap>
  DAX_EXEC_CONT_EXPORT
  typename boost::enable_if_c<N == FIRST_INDEX-1>::type
  CopyImpl(const Members<CopySig,CopyMap> &)
  {  }
};

}
} // namespace dax::internal

#endif //__dax__internal__Members_h
