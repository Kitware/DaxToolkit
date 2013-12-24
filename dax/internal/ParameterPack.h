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
//  Copyright 2013 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_internal_ParameterPack_h
#define __dax_internal_ParameterPack_h

#define DAX_PARAMETER_PACK_BUILD

#include <dax/Types.h>

#define DAX_MAX_PARAMETER_SIZE 10

#define BOOST_FUSION_INVOKE_PROCEDURE_MAX_ARITY DAX_MAX_PARAMETER_SIZE
#define FUSION_MAX_VECTOR_SIZE DAX_MAX_PARAMETER_SIZE

#include <boost/static_assert.hpp>

#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/utility/enable_if.hpp>

#include <boost/mpl/if.hpp>
#include <boost/mpl/assert.hpp>

#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/inc.hpp>

#ifdef DAX_USE_VARIADIC_TEMPLATE

#define _dax_pp_T___             T___...
#define _dax_pp_typename___T     typename ...T___
// Fake so that always applicable
#define _dax_pp_sizeof___T       1
#define _dax_pp_comma            ,
#define _dax_pp_params___(x)     T___... x
#define _dax_pp_args___(x)       x...

#define _dax_pp_params_ref___(x) T___&... x

#define _dax_pp_T___all          _dax_pp_T___
#define _dax_pp_typename___T_all _dax_pp_typename___T

#else //DAX_USE_VARIADIC_TEMPLATE

#include <dax/internal/ParameterPackCxx03.h>
#define _dax_pp_T___all                 BOOST_PP_ENUM(DAX_MAX_PARAMETER_SIZE, __dax_pp_T___, notUsed)
#define __dax_pp_name___(x,i)           BOOST_PP_CAT(x,BOOST_PP_INC(i))
#define __dax_pp_T___(z,i,x)            __dax_pp_name___(T___,i)
#define _dax_pp_typename___T_all        BOOST_PP_ENUM(DAX_MAX_PARAMETER_SIZE, __dax_pp_typename___T, notUsed)
#define __dax_pp_typename___T(z,i,x)    typename __dax_pp_T___(z,i,x) = dax::internal::detail::NullParam

#define _dax_pp_params_ref___(x)        _dax_pp_params___(&x)

#endif //DAX_USE_VARIADIC_TEMPLATE

/*
Quick rundown on what parameter pack can do for you, and how to use it.

Add a ParameterPack class to manage variable function arguments.

The new class dax::internal::ParameterPack is somewhat like boost::tuple
in that it has variable template arguments and holds an instance of each
of these types. However, this class is specifically meant to be used to
hold arguments to a function of unknown parameters. In addition to holding
the parameters, it allows you to modify the arguments (to modify the
behavior of the function) and to invoke an actual functor using the
arguments held in the class.


To make a parameter pack:
  dax::internal::ParameterPack<int,double,char> params =
      dax::internal::make_ParameterPack(1,2.5,'a');

To get the number of items:
  params.GetNumberOfParameters()

To get an element:
  params.GetArgument< 1 >()

  or

  dax::interal::ParameterPackGetArgument< 1 >( params )

Argument indexing in a parameter pack is 1 based, NOT 0 based

To change the value of an argument :
  params.SetArgument<1>( 100 )

  or

  dax::interal::ParameterPackSetArgument< 1 >( params, 100 )

To create a new parameter pack with a extra argument appended to the end:
  params.Append<std::string>(std::string("New Arg"))

To create a new parameter pack with an argument replaced:
  params.Replace<1>(std::string("new fisrt argument"))


To Invoke a function in the dax control space with the parameter pack, that
has a void return value :
  params.InvokeCont( Functor() )

InvokeCont also supports an optional transform functor. Please Read
the documentation on InvokeCont to see how to use it.

To Invoke a function in the dax exec space with the parameter pack, that
has a void return value :
  params.InvokeExec( Functor() )

InvokeExec also supports an optional transform functor. Please Read
the documentation on InvokeExec to see how to use it.

To Invoke a function in the dax exec space with the parameter pack, that
has a void return value :
  params.InvokeExec( Functor() )

To Invoke a function that has a return type you will need to call one of the
following functions
  dax::internal::ParameterPackInvokeWithReturnCont< ReturnType >(Functor(), params)

  dax::internal::ParameterPackInvokeWithReturnExec< ReturnType >(Functor(), params)

*/

namespace dax {
namespace internal {

/// Used to overload the behavior of a constructor to copy values from a given
/// \c ParameterPack.
///
struct ParameterPackCopyTag {  };

/// Used to overload the behavior of a constructor to initialize all values in
/// a \c ParameterPack with a given object.
///
struct ParameterPackInitialArgumentTag {  };


/// Used to overload a \c ParameterPack method that is exported only in the
/// control environment.
///
struct ParameterPackContTag {  };

/// Used to overload a \c ParameterPack method that is exported in both the
/// control and exection environments.
///
struct ParameterPackExecContTag {  };

/// Used as the default transform for ParameterPack::InvokeCont.
///
struct IdentityFunctorCont
{
  template<typename T>
  struct ReturnType {
    typedef T type;
  };

  template<typename T>
  DAX_CONT_EXPORT
  T &operator()(T &x) const { return x; }
};

/// Used as the default transform for ParameterPack::InvokeExec.
struct IdentityFunctorExec
{
  template<typename T>
  struct ReturnType {
    typedef T type;
  };

  template<typename T>
  DAX_EXEC_EXPORT
  T &operator()(T &x) const { return x; }
};

namespace detail {

/// Placeholder for an unused parameter
struct NullParam { };

/// If you encounter this (partial) type, then you have probably given
/// ParameterPack::Parameter<Index> an invalid Index.
template<typename T> struct InvalidParameterPackType;

template<typename Signature>
struct ParameterPackFirstArgument;

template<typename Signature>
struct ParameterPackRemainingArguments;

template<typename ParameterPackImplType>
struct ParameterPackImplAccess;

template<int Index, typename ParameterPackImplType>
struct ParameterPackImplAtIndex;

template<typename Function,
         typename ExtractedArgSignature,
         typename Transform,
         typename RemainingArgsType>
struct ParameterPackDoInvokeContImpl;

template<typename Function,
         typename Transform,
         typename ParameterPackImplType>
DAX_CONT_EXPORT
void ParameterPackDoInvokeCont(Function &f,
                               const Transform &transform,
                               ParameterPackImplType &params)
{
  ParameterPackDoInvokeContImpl<
      Function,
      void(),
      Transform,
      ParameterPackImplType>
    implementation;
  implementation(f, transform, params);
}

template<typename Function,
         typename ExtractedArgSignature,
         typename Transform,
         typename RemainingArgsType>
struct ParameterPackDoInvokeExecImpl;

template<typename Function,
         typename Transform,
         typename ParameterPackImplType>
DAX_EXEC_EXPORT
void ParameterPackDoInvokeExec(Function &f,
                               const Transform &transform,
                               ParameterPackImplType &params)
{
  ParameterPackDoInvokeExecImpl<
      Function,
      void(),
      Transform,
      ParameterPackImplType>
    implementation;
  implementation(f, transform, params);
}

/// Implementation class of a ParameterPack. Uses a lisp-like cons construction
/// to build a list of types, but is templated on a function signature to make
/// errors more readable.
///
template<typename Signature>
class ParameterPackImpl
{
private:
  typedef typename ParameterPackFirstArgument<Signature>::type CarType;
  typedef ParameterPackImpl<
      typename ParameterPackRemainingArguments<Signature>::type> CdrType;

  typedef ParameterPackImpl<Signature> ThisType;

  template<typename ParameterPackImplType>
  friend struct ParameterPackImplAccess;

  // If there is a compile error here, then a parameter was incorrectly set to
  // the NullParam type.
  BOOST_MPL_ASSERT_NOT(( boost::is_same<CarType,NullParam> ));

public:
  DAX_EXEC_CONT_EXPORT
  ParameterPackImpl() {  }

  DAX_EXEC_CONT_EXPORT
  ParameterPackImpl(CarType car, const CdrType &cdr)
    : Car(car), Cdr(cdr) {  }

  DAX_EXEC_CONT_EXPORT
  ParameterPackImpl(const ThisType &src)
    : Car(src.Car), Cdr(src.Cdr) {  }

  DAX_CONT_EXPORT
  ParameterPackImpl(CarType car, const CdrType &cdr, ParameterPackContTag)
    : Car(car), Cdr(cdr) {  }

  template<typename SrcSignature>
  DAX_CONT_EXPORT
  ParameterPackImpl(const ParameterPackImpl<SrcSignature> &src,
                    ParameterPackCopyTag,
                    ParameterPackContTag)
    : Car(ParameterPackImplAccess<ParameterPackImpl<SrcSignature> >
            ::GetFirstArgument(src)),
      Cdr(ParameterPackImplAccess<ParameterPackImpl<SrcSignature> >
            ::GetCdr(src),
          ParameterPackCopyTag(),
          ParameterPackContTag())
  {  }
  template<typename SrcSignature>
  DAX_EXEC_CONT_EXPORT
  ParameterPackImpl(const ParameterPackImpl<SrcSignature> &src,
                    ParameterPackCopyTag,
                    ParameterPackExecContTag)
    : Car(ParameterPackImplAccess<ParameterPackImpl<SrcSignature> >
            ::GetFirstArgument(src)),
      Cdr(ParameterPackImplAccess<ParameterPackImpl<SrcSignature> >
            ::GetCdr(src),
          ParameterPackCopyTag(),
          ParameterPackExecContTag())
  {  }

  template<typename SrcSignature>
  DAX_CONT_EXPORT
  ParameterPackImpl(ParameterPackImpl<SrcSignature> &src,
                    ParameterPackCopyTag,
                    ParameterPackContTag)
    : Car(ParameterPackImplAccess<ParameterPackImpl<SrcSignature> >
            ::GetFirstArgument(src)),
      Cdr(ParameterPackImplAccess<ParameterPackImpl<SrcSignature> >
            ::GetCdr(src),
          ParameterPackCopyTag(),
          ParameterPackContTag())
  {  }
  template<typename SrcSignature>
  DAX_EXEC_CONT_EXPORT
  ParameterPackImpl(ParameterPackImpl<SrcSignature> &src,
                    ParameterPackCopyTag,
                    ParameterPackExecContTag)
    : Car(ParameterPackImplAccess<ParameterPackImpl<SrcSignature> >
            ::GetFirstArgument(src)),
      Cdr(ParameterPackImplAccess<ParameterPackImpl<SrcSignature> >
            ::GetCdr(src),
          ParameterPackCopyTag(),
          ParameterPackExecContTag())
  {  }

  template<typename InitialParameter>
  DAX_EXEC_CONT_EXPORT
  ParameterPackImpl(InitialParameter &initial,
                    ParameterPackInitialArgumentTag,
                    ParameterPackExecContTag)
    : Car(initial),
      Cdr(initial,ParameterPackInitialArgumentTag(),ParameterPackExecContTag())
  {  }
  template<typename InitialParameter>
  DAX_CONT_EXPORT
  ParameterPackImpl(InitialParameter &initial,
                    ParameterPackInitialArgumentTag,
                    ParameterPackContTag)
    : Car(initial),
      Cdr(initial,ParameterPackInitialArgumentTag(),ParameterPackContTag())
  {  }

  DAX_EXEC_CONT_EXPORT
  ThisType &operator=(const ThisType &src) {
    this->Car = src.Car;
    this->Cdr = src.Cdr;
    return *this;
  }


private:
  CarType Car;
  CdrType Cdr;
};

template<>
class ParameterPackImpl<void()>
{
  typedef ParameterPackImpl<void()> ThisType;

public:
  DAX_EXEC_CONT_EXPORT
  ParameterPackImpl() {   }

  DAX_EXEC_CONT_EXPORT
  ParameterPackImpl(const ThisType &,
                    ParameterPackCopyTag,
                    ParameterPackExecContTag) {   }
  DAX_CONT_EXPORT
  ParameterPackImpl(const ThisType &,
                    ParameterPackCopyTag,
                    ParameterPackContTag) {   }

  template<typename T>
  DAX_EXEC_CONT_EXPORT
  ParameterPackImpl(const T &,
                    ParameterPackInitialArgumentTag,
                    ParameterPackExecContTag) {  }
  template<typename T>
  DAX_CONT_EXPORT
  ParameterPackImpl(const T &,
                    ParameterPackInitialArgumentTag,
                    ParameterPackContTag) {  }
};

typedef ParameterPackImpl<void()> ParameterPackImplNull;

template<typename ParameterPackImplType>
struct ParameterPackImplIsNull
{
  typedef typename
    boost::is_same<
      typename boost::remove_const<ParameterPackImplType>::type,
      ParameterPackImplNull>::type type;
};

template<typename ParameterPackType>
struct FindParameterPackImpl;

/// A cheating friend class that accesses the private members of ParameterPack.
/// It is necessary to access private members outside of ParameterPackImpl to
/// implement some of the features of variable arguments.
template<typename ParameterPackImplType>
struct ParameterPackImplAccess
{
  typedef typename ParameterPackImplType::CarType CarType;
  typedef typename ParameterPackImplType::CdrType CdrType;

  DAX_EXEC_CONT_EXPORT
  static void SetFirstArgument(
      ParameterPackImplType &parameters,
      const typename boost::remove_reference<CarType>::type &arg){
    parameters.Car = arg;
  }

  DAX_EXEC_CONT_EXPORT
  static const typename boost::remove_reference<CarType>::type &
  GetFirstArgument(
      const typename boost::remove_const<ParameterPackImplType>::type &parameters) {
    return parameters.Car;
  }

  DAX_EXEC_CONT_EXPORT
  static typename boost::remove_reference<CarType>::type &
  GetFirstArgument(
      typename boost::remove_const<ParameterPackImplType>::type &parameters) {
    return parameters.Car;
  }

  DAX_EXEC_CONT_EXPORT
  static const CdrType &GetCdr(
      const typename boost::remove_const<ParameterPackImplType>::type &parameters) {
    return parameters.Cdr;
  }
  DAX_EXEC_CONT_EXPORT
  static CdrType &GetCdr(
      typename boost::remove_const<ParameterPackImplType>::type &parameters) {
    return parameters.Cdr;
  }
};

/// This class provides the size (in number of parameters) of a
/// ParameterPackImpl instance.
///
template<typename ParameterPackImplType>
struct ParameterPackImplSize
{
  static const int NUM_PARAMETERS =
      ParameterPackImplSize<
        typename ParameterPackImplAccess<ParameterPackImplType>::CdrType>::
      NUM_PARAMETERS + 1;
};
template<>
struct ParameterPackImplSize<ParameterPackImplNull>
{
  static const int NUM_PARAMETERS = 0;
};

/// This class provides the implementation of all types and methods that
/// rely on dereferencing an index.  It is here because it requires template
/// specialization, which cannot be done as an internal class.
template<int Index, typename ParameterPackImplType>
struct ParameterPackImplAtIndex
{
  // These check for a valid index. If you get a compile error in these lines,
  // it is probably caused by an invalid Index template parameter for one of
  // the ParameterPack functions, methods, or classes. Remember that parameters
  // are indexed starting at 1.
  BOOST_STATIC_ASSERT(Index > 0);
  BOOST_STATIC_ASSERT(
      Index <= ParameterPackImplSize<ParameterPackImplType>::NUM_PARAMETERS);

private:
  typedef ParameterPackImplAccess<ParameterPackImplType> Access;

  typedef ParameterPackImplAtIndex<Index-1, typename Access::CdrType>
    ShiftedParameterPackImplAtIndex;
public:
  typedef typename ShiftedParameterPackImplAtIndex::CarType CarType;
  typedef typename ShiftedParameterPackImplAtIndex::CdrType CdrType;

  DAX_EXEC_CONT_EXPORT
  static void SetArgument(
      ParameterPackImplType &parameters,
      const typename boost::remove_reference<CarType>::type &arg)
  {
    ShiftedParameterPackImplAtIndex::
        SetArgument(Access::GetCdr(parameters), arg);
  }

  DAX_EXEC_CONT_EXPORT
  static const typename boost::remove_reference<CarType>::type &
  GetArgument(const ParameterPackImplType &parameters)
  {
    return ShiftedParameterPackImplAtIndex::
        GetArgument(Access::GetCdr(parameters));
  }
  DAX_EXEC_CONT_EXPORT
  static typename boost::remove_reference<CarType>::type &
  GetArgument(ParameterPackImplType &parameters)
  {
    return ShiftedParameterPackImplAtIndex::
        GetArgument(Access::GetCdr(parameters));
  }
};
template<typename ParameterPackImplType>
struct ParameterPackImplAtIndex<1, ParameterPackImplType>
{
private:
  typedef ParameterPackImplAccess<ParameterPackImplType> Access;
public:
  typedef typename Access::CarType CarType;
  typedef typename Access::CdrType CdrType;

  DAX_EXEC_CONT_EXPORT
  static void SetArgument(
      ParameterPackImplType &parameters,
      const typename boost::remove_reference<CarType>::type &arg) {
    Access::SetFirstArgument(parameters, arg);
  }
  DAX_EXEC_CONT_EXPORT
  static const typename boost::remove_reference<CarType>::type &
  GetArgument(const ParameterPackImplType &parameters) {
    return Access::GetFirstArgument(parameters);
  }
  DAX_EXEC_CONT_EXPORT
  static typename boost::remove_reference<CarType>::type &
  GetArgument(ParameterPackImplType &parameters) {
    return Access::GetFirstArgument(parameters);
  }
};

template<typename ParameterPackType,
         typename AppendType>
struct ParameterPackAppendType;

template<typename PrependType,
         typename ParameterPackType>
struct ParameterPackPrependType;

template<typename NewType, typename ParameterPackType>
struct PPackPrepend;

template<typename NewType, int Index, typename ParameterPackType>
struct PPackReplace;

//template<int Start, int End>
//struct CopyArgumentsImpl
//{
//  template<typename DestType, typename SrcType>
//  DAX_EXEC_CONT_EXPORT
//  void operator()(DestType &dest, const SrcType &src) {
//    CopyArgumentsImpl<Start,End-1>()(dest, src);
//    dest.template SetArgument<End-1>(src.template GetArgument<End-1>());
//  }
//};

//template<int Index>
//struct CopyArgumentsImpl<Index, Index>
//{
//  template<typename DestType, typename SrcType>
//  DAX_EXEC_CONT_EXPORT
//  void operator()(DestType &daxNotUsed(dest), const SrcType &daxNotUsed(src)) {
//    // Nothing left to copy.
//  }
//};

} // namespace detail (in dax::internal)

/// \brief Sets the argument at \c Index of the given \c ParameterPack.
///
/// The statement
/// \code{.cpp}
/// ParameterPackSetArgument<Index>(params, arg);
/// \endcode
/// is equivalent to either
/// \code{.cpp}
/// params.template SetArgument<Index>(arg);
/// \endcode
/// or
/// \code{.cpp}
/// params.SetArgument<Index>(arg);
/// \endcode
/// depending on the context that it is used. Using this function allows
/// you not not worry about whether the \c template keyword is necessary.
///
template<int Index, typename ParameterPackType, typename ParameterType>
DAX_EXEC_CONT_EXPORT
void ParameterPackSetArgument(ParameterPackType &params,
                              const ParameterType &arg)
{
  params.template SetArgument<Index>(arg);
}

/// \brief Returns the argument at \c Index of the given \c ParameterPack.
///
/// The statement
/// \code{.cpp}
/// value = ParameterPackGetArgument<Index>(params);
/// \endcode
/// is equivalent to either
/// \code{.cpp}
/// value = params.template GetArgument<Index>();
/// \endcode
/// or
/// \code{.cpp}
/// value = params.GetArgument<Index>();
/// \endcode
/// depending on the context that it is used. Using this function allows
/// you not not worry about whether the \c template keyword is necessary.
///
template<int Index, typename ParameterPackType>
DAX_EXEC_CONT_EXPORT
typename ParameterPackType::template Parameter<Index>::type
ParameterPackGetArgument(const ParameterPackType &params)
{
  return params.template GetArgument<Index>();
}

/// \brief Holds an arbitrary set of parameters in a single class.
///
/// To make using Dax easier for the end user developer, the
/// dax::cont::Dispatcher*::Invoke() method takes an arbitrary amount of
/// arguments that get transformed and swizzled into arguments and return value
/// for a worklet operator. In between these two invocations in a complicated
/// series of transformations and operations that can occur.
///
/// Supporting arbitrary function and template arguments is difficult and
/// really requires seperate implementations for ANSI and C++11 versions of
/// compilers. Thus, variatic template arguments are, at this point in time,
/// something to be avoided when possible. The intention of \c ParameterPack is
/// to collect most of the variatic template code into one place. The
/// ParameterPack template class takes a variable number of arguments that are
/// intended to match the parameters of some function. This means that all
/// arguments can be passed around in a single object so that objects and
/// functions dealing with these variadic parameters can be templated on a
/// single type (the type of ParameterPack).
///
/// Note that the indexing of the parameters in a \c ParameterPack starts at 1.
/// Although this is uncommon in C++, it matches better the parameter indexing
/// for other classes that deal with signatures for whole functions that use
/// the 0 index for the return value.
///
/// The \c ParameterPack contains several ways to invoke a functor whose
/// parameters match those of the parameter pack. This allows you to complete
/// the transition of calling an arbitrary function (like a worklet).
///
template<_dax_pp_typename___T_all>
class ParameterPack
    : public detail::FindParameterPackImpl<ParameterPack<_dax_pp_T___all> >::type
{
private:
  typedef typename
    detail::FindParameterPackImpl<ParameterPack<_dax_pp_T___all> >::type
      ImplementationType;
  typedef ImplementationType Superclass;
  typedef ParameterPack<_dax_pp_T___all> ThisType;
public:
  DAX_EXEC_CONT_EXPORT
  ParameterPack() {  }

  DAX_EXEC_CONT_EXPORT
  ParameterPack(const ThisType &src)
    : Superclass(src) {  }

  /// \brief Copy data from another \c ParameterPack
  ///
  /// The first argument is a source \c ParameterPack to copy from. The
  /// parameter types of the \c src \c ParameterPack do not have to be the
  /// exact same type as this object, but must be able to work in a copy
  /// constructor. The second argument is an instance of the \c
  /// ParamterPackCopyTag.
  ///
  template<typename SrcSignature>
  DAX_EXEC_CONT_EXPORT
  ParameterPack(const detail::ParameterPackImpl<SrcSignature> &src,
                ParameterPackCopyTag,
                ParameterPackExecContTag)
    : Superclass(src, ParameterPackCopyTag(), ParameterPackExecContTag()) {  }
  template<typename SrcSignature>
  DAX_CONT_EXPORT
  ParameterPack(const detail::ParameterPackImpl<SrcSignature> &src,
                ParameterPackCopyTag,
                ParameterPackContTag)
    : Superclass(src, ParameterPackCopyTag(), ParameterPackContTag()) {  }

  /// \brief Copy data from another \c ParameterPack
  ///
  /// The first argument is a source \c ParameterPack to copy from. The
  /// parameter types of the \c src \c ParameterPack do not have to be the
  /// exact same type as this object, but must be able to work in a copy
  /// constructor. The second argument is an instance of the \c
  /// ParamterPackCopyTag.
  ///
  template<typename SrcSignature>
  DAX_EXEC_CONT_EXPORT
  ParameterPack(detail::ParameterPackImpl<SrcSignature> &src,
                ParameterPackCopyTag,
                ParameterPackExecContTag)
    : Superclass(src, ParameterPackCopyTag(), ParameterPackExecContTag()) {  }
  template<typename SrcSignature>
  DAX_CONT_EXPORT
  ParameterPack(detail::ParameterPackImpl<SrcSignature> &src,
                ParameterPackCopyTag,
                ParameterPackContTag)
    : Superclass(src, ParameterPackCopyTag(), ParameterPackContTag()) {  }

  /// \brief Initialize all the parameters with the given argument.
  ///
  /// The first argument is past to the constructors of all arguments held in
  /// this \c ParameterPack. The second argument is an instance of the \c
  /// ParameterPackInitialArgumentTag.
  ///
  template<typename InitialArgumentType>
  DAX_EXEC_CONT_EXPORT
  ParameterPack(InitialArgumentType &initial,
                ParameterPackInitialArgumentTag,
                ParameterPackExecContTag)
    : Superclass(initial,
                 ParameterPackInitialArgumentTag(),
                 ParameterPackExecContTag())
  {  }
  template<typename InitialArgumentType>
  DAX_CONT_EXPORT
  ParameterPack(InitialArgumentType &initial,
                ParameterPackInitialArgumentTag,
                ParameterPackContTag)
    : Superclass(initial,
                 ParameterPackInitialArgumentTag(),
                 ParameterPackContTag())
  {  }

  DAX_EXEC_CONT_EXPORT
  ThisType &operator=(const ThisType &src)
  {
    this->Superclass::operator=(src);
    return *this;
  }

  /// \brief Provides type information about a particular parameter.
  ///
  /// The templated \c Parameter subclass provides type information about a
  /// particular parmater specified by the template parameter \c Index. The \c
  /// Parameter subclass contains a typedef named \c type set to the given
  /// parameter type.
  ///
  template<int Index>
  struct Parameter {
    /// \brief Type of the parameter at \c Index.
    ///
    typedef typename
        detail::ParameterPackImplAtIndex<Index, Superclass>::CarType type;
  };

  /// The number of parameters in this \c ParameterPack type.
  ///
  const static int NUM_PARAMETERS =
      detail::ParameterPackImplSize<Superclass>::NUM_PARAMETERS;

  /// Returns the number of parameters (and argument values) held in this
  /// \c ParameterPack. The return value is the same as \c NUM_PARAMETERS.
  ///
  DAX_EXEC_CONT_EXPORT
  dax::Id GetNumberOfParameters() const {
    return ParameterPack::NUM_PARAMETERS;
  }

  /// Sets the argument associated with the template parameter \c Index to the
  /// given \c argument.
  template<int Index>
  DAX_EXEC_CONT_EXPORT
  void SetArgument(const typename boost::remove_reference<
                     typename Parameter<Index>::type>::type &argument)
  {
    detail::ParameterPackImplAtIndex<Index, Superclass>::
        SetArgument(*this, argument);
  }

  /// Returns the argument associated with the template parameter \c Index.
  ///
  template<int Index>
  DAX_EXEC_CONT_EXPORT
  const typename boost::remove_reference<typename Parameter<Index>::type>::type&
  GetArgument() const
  {
    return detail::ParameterPackImplAtIndex<Index, Superclass>::
        GetArgument(*this);
  }
  template<int Index>
  DAX_EXEC_CONT_EXPORT
  typename boost::remove_reference<typename Parameter<Index>::type>::type &
  GetArgument()
  {
    return detail::ParameterPackImplAtIndex<Index, Superclass>::
        GetArgument(*this);
  }

  /// Invoke a function \c f using the arguments stored in this \c
  /// ParameterPack.
  ///
  /// An optional \c parameterTransform functor allows you to transform each
  /// argument before passing it to the function. In addition to an overloaded
  /// parenthesis operator, the \c Transform class must also have a templated
  /// subclass named \c ReturnType containing a typedef named \c type giving
  /// the return type for a given type. For example, the default \c Transform
  /// is an identity function that looks essentially like this.
  ///
  /// \code{.cpp}
  /// struct IdentityTransform
  /// {
  ///   template<typename T> struct ReturnType {
  ///     typedef T typedef;
  ///   };
  ///   template<typename T> T &operator()(T &x) const { return x; }
  /// };
  /// \endcode
  ///
  /// As another example, this \c Transform class converts a reference to an
  /// argument to a pointer to that argument.
  ///
  /// \code{.cpp}
  /// struct GetReferenceTransform
  /// {
  ///   template<typename T> struct ReturnType {
  ///     typedef const typename boost::remove_reference<T>::type *type;
  ///   };
  ///   template<typename T> T *operator()(T &x) const { return &x; }
  /// };
  /// \endcode
  ///
  template<typename Function, typename Transform>
  DAX_CONT_EXPORT
  void InvokeCont(Function &f, const Transform &parameterTransform) const
  {
    detail::ParameterPackDoInvokeCont(f, parameterTransform, *this);
  }
  template<typename Function, typename Transform>
  DAX_CONT_EXPORT
  void InvokeCont(Function &f, const Transform &parameterTransform)
  {
    detail::ParameterPackDoInvokeCont(f, parameterTransform, *this);
  }
  template<typename Function, typename Transform>
  DAX_CONT_EXPORT
  void InvokeCont(const Function &f, const Transform &parameterTransform) const
  {
    detail::ParameterPackDoInvokeCont(f, parameterTransform, *this);
  }
  template<typename Function, typename Transform>
  DAX_CONT_EXPORT
  void InvokeCont(const Function &f, const Transform &parameterTransform)
  {
    detail::ParameterPackDoInvokeCont(f, parameterTransform, *this);
  }
  template<typename Function>
  DAX_CONT_EXPORT
  void InvokeCont(Function &f) const
  {
    this->InvokeCont(f, IdentityFunctorCont());
  }
  template<typename Function>
  DAX_CONT_EXPORT
  void InvokeCont(Function &f)
  {
    this->InvokeCont(f, IdentityFunctorCont());
  }
  template<typename Function>
  DAX_CONT_EXPORT
  void InvokeCont(const Function &f) const
  {
    this->InvokeCont(f, IdentityFunctorCont());
  }
  template<typename Function>
  DAX_CONT_EXPORT
  void InvokeCont(const Function &f)
  {
    this->InvokeCont(f, IdentityFunctorCont());
  }
  template<typename Function, typename Transform>
  DAX_EXEC_EXPORT
  void InvokeExec(Function &f, const Transform &parameterTransform) const
  {
    detail::ParameterPackDoInvokeExec(f, parameterTransform, *this);
  }
  template<typename Function, typename Transform>
  DAX_EXEC_EXPORT
  void InvokeExec(Function &f, const Transform &parameterTransform)
  {
    detail::ParameterPackDoInvokeExec(f, parameterTransform, *this);
  }
  template<typename Function, typename Transform>
  DAX_EXEC_EXPORT
  void InvokeExec(const Function &f, const Transform &parameterTransform) const
  {
    detail::ParameterPackDoInvokeExec(f, parameterTransform, *this);
  }
  template<typename Function, typename Transform>
  DAX_EXEC_EXPORT
  void InvokeExec(const Function &f, const Transform &parameterTransform)
  {
    detail::ParameterPackDoInvokeExec(f, parameterTransform, *this);
  }
  template<typename Function>
  DAX_EXEC_EXPORT
  void InvokeExec(Function &f) const
  {
    this->InvokeExec(f, IdentityFunctorExec());
  }
  template<typename Function>
  DAX_EXEC_EXPORT
  void InvokeExec(Function &f)
  {
    this->InvokeExec(f, IdentityFunctorExec());
  }
  template<typename Function>
  DAX_EXEC_EXPORT
  void InvokeExec(const Function &f) const
  {
    this->InvokeExec(f, IdentityFunctorExec());
  }
  template<typename Function>
  DAX_EXEC_EXPORT
  void InvokeExec(const Function &f)
  {
    this->InvokeExec(f, IdentityFunctorExec());
  }

  /// \brief Append an argument to this \c ParameterPack.
  ///
  /// Returns a new \c ParameterPack where the first arguments are all the same
  /// as the ones in this \c ParameterPack and then \c newArg is added to the
  /// end of the argument list.
  ///
  /// The \c Append method is intended to pass further arguments to a called
  /// function without worrying about manipulation of the type list. \c Append
  /// method invocations can be chained together to specify multiple arguments
  /// to be appended. The following is a simple example of a method that
  /// derives some arrays from a set of keys and then passes this information
  /// to another method that presumably knows how to use them.
  ///
  /// \code{.cpp}
  /// template<typename ParameterPackType>
  /// void DoSomethingWithKeys(const ParameterPackType &arguments)
  /// {
  ///   typedef dax::cont::internal::DeviceAdapterAlgorithm<DeviceAdapterTag>
  ///       Algorithm;
  ///   typedef typename ParameterPackType::template Parameter<1>::type
  ///       KeysType;
  ///
  ///   KeysType keys = arguments::template Get<1>();
  ///
  ///   KeysType sortedKeys;
  ///   Algorithm::Copy(keys, sortedKeys);
  ///   Algorithm::Sort(sortedKeys);
  ///
  ///   KeysType uniqueKeys;
  ///   Algorithm::Copy(sortedKeys, uniqueKeys);
  ///   Algorithm::Unique(uniqueKeys);
  ///
  ///   DoSomethingWithSortedKeys(
  ///       arguments
  ///       .Append(sortedKeys)
  ///       .Append(uniqueKeys));
  /// }
  /// \endcode
  ///
  /// The \c Append method is only supported for the control environment.
  ///
  template<typename NewType>
  DAX_CONT_EXPORT
  typename detail::ParameterPackAppendType<ThisType, NewType>::type
  Append(NewType newArg) const {
    typedef detail::ParameterPackAppendType<ThisType, NewType> PPackAppendType;
    typename PPackAppendType::type appendedPack(
          PPackAppendType::ConstructImpl(*this, newArg),
          ParameterPackCopyTag(),
          ParameterPackContTag());
    return appendedPack;
  }

  /// \brief Replaces an argument in this \c ParameterPack
  ///
  /// Returns a new ParameterPack with all entries replaced except for the
  /// argument at the given index, which is replaced with \c newArg, which can
  /// be a different type.
  ///
  /// \code{.cpp}
  /// template<typename ParameterPackType>
  /// void MangleFirstArgument(const ParameterPackType &arguments)
  /// {
  ///   typedef typename ParameterPackType::template Parameter<1>::type
  ///       FirstArgType;
  ///   FirstArgType firstArg = arguments::template Get<1>();
  ///
  ///   // Derive new argumente mangledArg...
  ///
  ///   DoSomethingElse(arguments.template Replace<1>(mangledArg));
  /// }
  /// \endcode
  ///
  /// The \c Replace method is only supported for the control environment.
  ///
  template<int Index, typename NewType>
  DAX_CONT_EXPORT
  typename detail::PPackReplace<NewType, Index, ThisType>::type
  Replace(NewType newArg) const {
    BOOST_STATIC_ASSERT(Index > 0);
    BOOST_STATIC_ASSERT(Index <= NUM_PARAMETERS);
    typedef detail::PPackReplace<NewType, Index, ThisType> PPackReplaceType;
    typename PPackReplaceType::type replacedPack(
          PPackReplaceType::ConstructImpl(newArg, *this),
          ParameterPackCopyTag(),
          ParameterPackContTag());
    return replacedPack;
  }
};

namespace detail {

template<>
struct FindParameterPackImpl<ParameterPack<> >
{
  typedef ParameterPackImplNull type;

  DAX_EXEC_CONT_EXPORT
  static type Construct() { return type(); }
};

template<typename AppendType>
struct ParameterPackAppendType<ParameterPack<>, AppendType>
{
  typedef ParameterPack<AppendType> type;

private:
  typedef detail::ParameterPackImpl<void(AppendType)> _implType;

public:
  DAX_CONT_EXPORT
  static _implType
  ConstructImpl(ParameterPackImplNull, AppendType toAppend)
  {
    return _implType(toAppend, ParameterPackImplNull());
  }
};

} // namespace detail

/// Given an instance of the \c ParameterPack template and an (optional) return
/// type, provides the type for the function that accepts those arguments and
/// returns a value of that type.  The appropriate function type is available
/// as a typedef in this class named \c type.
///
template<typename ParameterPackType, typename ReturnType = void>
struct ParameterPackToSignature
#ifdef DAX_DOXYGEN_ONLY
{
  /// The type signature for a function with the given arguments and return
  /// type.
  ///
  typedef ReturnType type(parameters);
}
#endif
;

/// Given a type for a function, provides the type for a ParameterPack that
/// matches the parameters of that function.  The appropriate \c ParameterPack
/// type is available as a typedef in this class named \c type.
///
template<typename Signature>
struct SignatureToParameterPack
#ifdef DAX_DOXYGEN_ONLY
{
  /// The type signature for a \c ParameterPack that matches the parameters of
  /// the template's function \c Signature type.
  ///
  typedef ParameterPack<parameters> type;
}
#endif
;

/// Invokes the given function with the arguments stored in the given
/// \c ParameterPack.
///
/// An optional \c Transform functor will be called on each argument before
/// being passed on to \c f.  See the ParameterPack::Invoke functions for
/// a description of the requirements for the \c Transform type.
///
template<typename Function, typename ParameterPackType, typename Transform>
DAX_CONT_EXPORT
void ParameterPackInvokeCont(Function &f,
                             ParameterPackType &params,
                             const Transform &parameterTransform)
{
  params.InvokeCont(f, parameterTransform);
}
template<typename Function, typename ParameterPackType, typename Transform>
DAX_CONT_EXPORT
void ParameterPackInvokeCont(const Function &f,
                             ParameterPackType &params,
                             const Transform &parameterTransform)
{
  params.InvokeCont(f, parameterTransform);
}
template<typename Function, typename ParameterPackType>
DAX_CONT_EXPORT
void ParameterPackInvokeCont(Function &f,
                             ParameterPackType &params)
{
  ParameterPackInvokeCont(f, params, IdentityFunctorCont());
}
template<typename Function, typename ParameterPackType>
DAX_CONT_EXPORT
void ParameterPackInvokeCont(const Function &f,
                             ParameterPackType &params)
{
  ParameterPackInvokeCont(f, params, IdentityFunctorCont());
}

template<typename Function, typename ParameterPackType, typename Transform>
DAX_EXEC_EXPORT
void ParameterPackInvokeExec(Function &f,
                             ParameterPackType &params,
                             const Transform &parameterTransform)
{
  params.InvokeExec(f, parameterTransform);
}
template<typename Function, typename ParameterPackType, typename Transform>
DAX_EXEC_EXPORT
void ParameterPackInvokeExec(const Function &f,
                             ParameterPackType &params,
                             const Transform &parameterTransform)
{
  params.InvokeExec(f, parameterTransform);
}
template<typename Function, typename ParameterPackType>
DAX_EXEC_EXPORT
void ParameterPackInvokeExec(Function &f,
                             ParameterPackType &params)
{
  ParameterPackInvokeExec(f, params, IdentityFunctorExec());
}
template<typename Function, typename ParameterPackType>
DAX_EXEC_EXPORT
void ParameterPackInvokeExec(const Function &f,
                             ParameterPackType &params)
{
  ParameterPackInvokeExec(f, params, IdentityFunctorExec());
}

namespace detail {

// Given a ParameterPack and a Transform like the type passed to an Invoke
// method, get a ParameterPack with the transformed arguments.
template<typename ParameterPackType, typename Transform>
struct ParameterPackTransform;

template<typename Transform>
struct ParameterPackTransform<ParameterPack<>, Transform>
{
  typedef ParameterPack<> type;
};

template<typename ReturnType, typename FunctionType>
class ParameterPackReturnFunctorContBase
{
  typedef ParameterPackReturnFunctorContBase<ReturnType, FunctionType> ThisType;

public:
  DAX_CONT_EXPORT
  ParameterPackReturnFunctorContBase(const FunctionType &f)
    : Function(f)
  {  }

  DAX_CONT_EXPORT
  ReturnType GetReturnValue() const {
    return this->ReturnValue;
  }

protected:
  DAX_CONT_EXPORT
  void RecordReturnValue(const ReturnType &returnValue) {
    this->ReturnValue = returnValue;
  }

private:
  DAX_CONT_EXPORT
  ParameterPackReturnFunctorContBase(const ThisType &); // Not implemented.
  DAX_CONT_EXPORT
  void operator()(const ThisType &); // Not implemented.

  ReturnType ReturnValue;

protected:
  FunctionType Function;
};

// Special implementation to return a reference.
template<typename ReturnType, typename FunctionType>
class ParameterPackReturnFunctorContBase<ReturnType &, FunctionType>
    : protected ParameterPackReturnFunctorContBase<ReturnType *, FunctionType>
{
  typedef ParameterPackReturnFunctorContBase<ReturnType *, FunctionType>
      Superclass;
public:
  DAX_CONT_EXPORT
  ParameterPackReturnFunctorContBase(const FunctionType &f)
    : Superclass(f) {  }

  DAX_CONT_EXPORT
  ReturnType &GetReturnValue() const {
    return *this->Superclass::GetReturnValue();
  }

protected:
  DAX_CONT_EXPORT
  void RecordReturnValue(ReturnType &returnValue) {
    this->Superclass::RecordReturnValue(&returnValue);
  }
};

template<typename ReturnType, typename FunctionType>
class ParameterPackReturnFunctorExecBase
{
  typedef ParameterPackReturnFunctorExecBase<ReturnType, FunctionType> ThisType;

public:
  DAX_EXEC_EXPORT
  ParameterPackReturnFunctorExecBase(const FunctionType &f)
    : Function(f)
  {  }

  DAX_EXEC_EXPORT
  ReturnType GetReturnValue() const {
    return this->ReturnValue;
  }

protected:
  DAX_EXEC_EXPORT
  void RecordReturnValue(const ReturnType &returnValue) {
    this->ReturnValue = returnValue;
  }

private:
  DAX_CONT_EXPORT
  ParameterPackReturnFunctorExecBase(const ThisType &); // Not implemented.
  DAX_CONT_EXPORT
  void operator()(const ThisType &); // Not implemented.

  ReturnType ReturnValue;

protected:
  FunctionType Function;
};

// Special implementation to return a reference.
template<typename ReturnType, typename FunctionType>
class ParameterPackReturnFunctorExecBase<ReturnType &, FunctionType>
    : protected ParameterPackReturnFunctorExecBase<ReturnType *, FunctionType>
{
  typedef ParameterPackReturnFunctorExecBase<ReturnType *, FunctionType>
      Superclass;
public:
  DAX_EXEC_EXPORT
  ParameterPackReturnFunctorExecBase(const FunctionType &f)
    : Superclass(f) {  }

  DAX_EXEC_EXPORT
  ReturnType &GetReturnValue() const {
    return *this->Superclass::GetReturnValue();
  }

protected:
  DAX_EXEC_EXPORT
  void RecordReturnValue(ReturnType &returnValue) {
    this->Superclass::RecordReturnValue(&returnValue);
  }
};

template<typename Signature, typename Function>
class ParameterPackReturnFunctorCont;

template<typename Signature, typename Function>
class ParameterPackReturnFunctorExec;

} // namespace detail (in dax::internal)

/// Invokes the given function with the arguments stored in the given
/// \c ParameterPack.  The functor is assumed to return a value of type
/// \c ReturnType, and this value will be returned from this function.
///
/// An optional \c Transform functor will be called on each argument before
/// being passed on to \c f.  See the ParameterPack::Invoke functions for
/// a description of the requirements for the \c Transform type.
///
template<typename ReturnType,
         typename Function,
         typename ParameterPackType,
         typename Transform>
DAX_CONT_EXPORT
ReturnType ParameterPackInvokeWithReturnCont(
    Function &f,
    ParameterPackType &params,
    const Transform &parameterTransform)
{
  detail::ParameterPackReturnFunctorCont<
      typename ParameterPackToSignature<
        typename detail::ParameterPackTransform<ParameterPackType, Transform>::type,
        ReturnType>::type,
      Function> functor(f);
  ParameterPackInvokeCont(functor, params, parameterTransform);
  return functor.GetReturnValue();
}
template<typename ReturnType,
         typename Function,
         typename ParameterPackType,
         typename Transform>
DAX_CONT_EXPORT
ReturnType ParameterPackInvokeWithReturnCont(
    const Function &f,
    ParameterPackType &params,
    const Transform &parameterTransform)
{
  detail::ParameterPackReturnFunctorCont<
      typename ParameterPackToSignature<
        typename detail::ParameterPackTransform<ParameterPackType, Transform>::type,
        ReturnType>::type,
      const Function> functor(f);
  ParameterPackInvokeCont(functor, params, parameterTransform);
  return functor.GetReturnValue();
}
template<typename ReturnType,
         typename Function,
         typename ParameterPackType>
DAX_CONT_EXPORT
ReturnType ParameterPackInvokeWithReturnCont(Function &f,
                                             ParameterPackType &params)
{
  return ParameterPackInvokeWithReturnCont<ReturnType>(
        f, params, IdentityFunctorCont());
}
template<typename ReturnType,
         typename Function,
         typename ParameterPackType>
DAX_CONT_EXPORT
ReturnType ParameterPackInvokeWithReturnCont(const Function &f,
                                             ParameterPackType &params)
{
  return ParameterPackInvokeWithReturnCont<ReturnType>(
        f, params, IdentityFunctorCont());
}

template<typename ReturnType,
         typename Function,
         typename ParameterPackType,
         typename Transform>
DAX_EXEC_EXPORT
ReturnType ParameterPackInvokeWithReturnExec(
    Function &f,
    ParameterPackType &params,
    const Transform &parameterTransform)
{
  detail::ParameterPackReturnFunctorExec<
      typename ParameterPackToSignature<
        typename detail::ParameterPackTransform<ParameterPackType, Transform>::type,
        ReturnType>::type,
      Function> functor(f);
  ParameterPackInvokeExec(functor, params, parameterTransform);
  return functor.GetReturnValue();
}
template<typename ReturnType,
         typename Function,
         typename ParameterPackType,
         typename Transform>
DAX_EXEC_EXPORT
ReturnType ParameterPackInvokeWithReturnExec(
    const Function &f,
    ParameterPackType &params,
    const Transform &parameterTransform)
{
  detail::ParameterPackReturnFunctorExec<
      typename ParameterPackToSignature<
        typename detail::ParameterPackTransform<ParameterPackType, Transform>::type,
        ReturnType>::type,
      const Function> functor(f);
  ParameterPackInvokeExec(functor, params, parameterTransform);
  return functor.GetReturnValue();
}
template<typename ReturnType,
         typename Function,
         typename ParameterPackType>
DAX_EXEC_EXPORT
ReturnType ParameterPackInvokeWithReturnExec(Function &f,
                                             ParameterPackType &params)
{
  return ParameterPackInvokeWithReturnExec<ReturnType>(
        f, params, IdentityFunctorExec());
}
template<typename ReturnType,
         typename Function,
         typename ParameterPackType>
DAX_EXEC_EXPORT
ReturnType ParameterPackInvokeWithReturnExec(const Function &f,
                                             ParameterPackType &params)
{
  return ParameterPackInvokeWithReturnExec<ReturnType>(
        f, params, IdentityFunctorExec());
}

namespace detail {

template<typename Signature>
struct ParameterPackConstructFunctorCont;

template<typename Signature>
struct ParameterPackConstructFunctorExec;

} // namespace detail

/// Constructs an object of type \c ObjectToConstruct by passing the
/// arguments in \c params to the object's constructor and returns this
/// newly created object.
///
/// An optional \c Transform functor will be called on each argument before
/// being passed on to \c f.  See the ParameterPack::Invoke functions for
/// a description of the requirements for the \c Transform type.
///
template<typename ObjectToConstruct,
         typename ParameterPackType,
         typename Transform>
DAX_CONT_EXPORT
ObjectToConstruct ParameterPackConstructCont(
    ParameterPackType &params,
    const Transform &parameterTransform)
{
  detail::ParameterPackConstructFunctorCont<
      typename ParameterPackToSignature<
        typename detail::ParameterPackTransform<ParameterPackType, Transform>::type,
        ObjectToConstruct>::type>
      functor;
  return ParameterPackInvokeWithReturnCont<ObjectToConstruct>(
        functor, params, parameterTransform);
}
template<typename ObjectToConstruct, typename ParameterPackType>
DAX_CONT_EXPORT
ObjectToConstruct ParameterPackConstructCont(ParameterPackType &params)
{
  return ParameterPackConstructCont<ObjectToConstruct>(
        params, IdentityFunctorCont());
}

template<typename ObjectToConstruct,
         typename ParameterPackType,
         typename Transform>
DAX_EXEC_EXPORT
ObjectToConstruct ParameterPackConstructExec(
    ParameterPackType &params,
    const Transform &parameterTransform)
{
  detail::ParameterPackConstructFunctorExec<
      typename ParameterPackToSignature<
        typename detail::ParameterPackTransform<ParameterPackType, Transform>::type,
        ObjectToConstruct>::type>
      functor;
  return ParameterPackInvokeWithReturnExec<ObjectToConstruct>(
        functor, params, parameterTransform);
}
template<typename ObjectToConstruct, typename ParameterPackType>
DAX_EXEC_EXPORT
ObjectToConstruct ParameterPackConstructExec(ParameterPackType &params)
{
  return ParameterPackConstructExec<ObjectToConstruct>(
        params, IdentityFunctorExec());
}

#if defined(DAX_DOXYGEN_ONLY) || defined(DAX_USE_VARIADIC_TEMPLATE)
/// Convenience function that creates a \c ParameterPack with parameters of the
/// same types as the arguments to this function and initialized to the values
/// passed to this function. Only works in the control environment.
///
template<typename...T>
DAX_CONT_EXPORT
dax::internal::ParameterPack<T...>
make_ParameterPack(T...arguments);
#endif // DAX_DOXYGON_ONLY || DAX_USE_VARIADIC_TEMPLATE


}
} // namespace dax::internal

#ifdef DAX_USE_VARIADIC_TEMPLATE

#include <dax/internal/ParameterPackVariadic.h>

#else //DAX_USE_VARIADIC_TEMPLATE

#define BOOST_PP_ITERATION_PARAMS_1 (3, (1, BOOST_PP_INC(DAX_MAX_PARAMETER_SIZE), <dax/internal/ParameterPackVariadic.h>))
#include BOOST_PP_ITERATE()

#endif //DAX_USE_VARIADIC_TEMPLATE

#undef _dax_pp_T___all
#undef __dax_pp_T___
#undef _dax_pp_typename___T_all
#undef __dax_pp_typename___T

#undef DAX_PARAMETER_PACK_BUILD

#endif //__dax_internal_ParameterPack_h
