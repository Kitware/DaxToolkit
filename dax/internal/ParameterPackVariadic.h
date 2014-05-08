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

// This won't compile correctly in a test build, and that's OK since it is
// only allowed to be compiled within ParameterPack.h
#if !defined(DAX_TEST_HEADER_BUILD) || defined(DAX_PARAMETER_PACK_BUILD)

#ifndef DAX_PARAMETER_PACK_BUILD
#error This file should only be included within ParameterPack.h
#endif


namespace dax {
namespace internal {
namespace detail {

#if _dax_pp_sizeof___T < DAX_MAX_PARAMETER_SIZE

template<typename T1 _dax_pp_comma _dax_pp_typename___T>
struct ParameterPackFirstArgument<void(T1 _dax_pp_comma _dax_pp_T___)>
{
  typedef T1 type;
};

template<typename T1 _dax_pp_comma _dax_pp_typename___T>
struct ParameterPackRemainingArguments<void(T1 _dax_pp_comma _dax_pp_T___)>
{
  typedef void type(_dax_pp_T___);
};

template<typename FirstType _dax_pp_comma _dax_pp_typename___T>
struct FindParameterPackImpl<
    ParameterPack<FirstType _dax_pp_comma _dax_pp_T___> >
{
  typedef ParameterPackImpl<void(FirstType _dax_pp_comma _dax_pp_T___)> type;

  DAX_EXEC_CONT_EXPORT
  static type Construct(FirstType first _dax_pp_comma
                        _dax_pp_params___(rest))
  {
    return type(first,
                FindParameterPackImpl<ParameterPack<_dax_pp_T___> >::Construct(
                  _dax_pp_args___(rest)));
  }
};

#endif //_dax_pp_sizeof___T < DAX_MAX_PARAMETER_SIZE

#if _dax_pp_sizeof___T < DAX_MAX_PARAMETER_SIZE - 1

template<typename FirstType,
         _dax_pp_typename___T _dax_pp_comma
         typename AppendType>
struct ParameterPackAppendType<
    ParameterPack<FirstType _dax_pp_comma _dax_pp_T___>,AppendType>
{
  typedef ParameterPack<FirstType _dax_pp_comma _dax_pp_T___, AppendType> type;

private:
  typedef typename FindParameterPackImpl<type>::type _implType;
  typedef typename FindParameterPackImpl<
    ParameterPack<FirstType _dax_pp_comma _dax_pp_T___> >::type _implInputType;

public:
  DAX_CONT_EXPORT
  static _implType ConstructImpl(const _implInputType &originals,
                                 AppendType toAppend)
  {
    typedef ParameterPackImplAccess<_implInputType> Access;
    typedef ParameterPackAppendType<ParameterPack<_dax_pp_T___>, AppendType>
        PPackAppendRemainder;
    return _implType(
          Access::GetFirstArgument(originals),
          PPackAppendRemainder::ConstructImpl(Access::GetCdr(originals),
                                              toAppend),
          ParameterPackContTag());
  }
};

#endif //_dax_pp_sizeof___T < DAX_MAX_PARAMETER_SIZE - 1

#if _dax_pp_sizeof___T < DAX_MAX_PARAMETER_SIZE

template<typename PrependType _dax_pp_comma _dax_pp_typename___T>
struct ParameterPackPrependType<PrependType, ParameterPack<_dax_pp_T___> >
{
  typedef ParameterPack<PrependType _dax_pp_comma _dax_pp_T___> type;
};

#endif //_dax_pp_sizeof___T < DAX_MAX_PARAMETER_SIZE

#if _dax_pp_sizeof___T < DAX_MAX_PARAMETER_SIZE

template<typename FirstType,
         _dax_pp_typename___T _dax_pp_comma
         typename Transform>
struct ParameterPackTransform<
    ParameterPack<FirstType _dax_pp_comma _dax_pp_T___>,
    Transform>
{
  typedef typename
    ParameterPackPrependType<
      typename Transform::template ReturnType<FirstType>::type,
      typename ParameterPackTransform<ParameterPack<_dax_pp_T___>, Transform>::type
    >::type type;
};

#endif //_dax_pp_sizeof___T < DAX_MAX_PARAMETER_SIZE

// This overloaded function is here only to make errors and warnings more
// legible, particularly when the arguments given do not match the parameters
// of the functor.
template<typename Function _dax_pp_comma _dax_pp_typename___T>
DAX_EXEC_EXPORT
void ParameterPackCallFunctionCont(Function &f _dax_pp_comma
                                   _dax_pp_params___(arguments))
{
  // If you get a compiler error on the following line, it means we tried to
  // invoke a functor or worklet with the wrong arguments. Check the type of
  // FunctionType and look at its operator(). If the error is there is no
  // matched method in FunctionType::operator(), it probably means an invalid
  // argument is being passed. This is likely caused from bad arguments to
  // Dispatcher::Invoke.
  f(_dax_pp_args___(arguments));
}

#if _dax_pp_sizeof___T < DAX_MAX_PARAMETER_SIZE

template<typename Function,
         _dax_pp_typename___T _dax_pp_comma
         typename Transform,
         typename ParameterPackImplType>
struct ParameterPackDoInvokeContImpl<
    Function,
    void(_dax_pp_T___),
    Transform,
    ParameterPackImplType>
{
  DAX_CONT_EXPORT
  void operator()(Function &f,
                  _dax_pp_params___(arguments) _dax_pp_comma
                  const Transform &transform,
                  ParameterPackImplType &params)
  {
    typedef ParameterPackImplAccess<ParameterPackImplType> Access;
    typedef typename boost::mpl::if_<
        typename boost::is_const<ParameterPackImplType>::type,
        const typename Access::CdrType,
        typename Access::CdrType>::type RemainingArgumentsType;
    typedef typename boost::mpl::if_<
        typename boost::is_const<ParameterPackImplType>::type,
        const typename boost::remove_reference<typename Access::CarType>::type &,
        typename boost::remove_reference<typename Access::CarType>::type &>::type
          InputParameterType;
    typedef typename Transform::template ReturnType<InputParameterType>::type
        TransformedParameterType;
    // It is important to explicitly specify the template parameters. When
    // automatically determining the types, then reference types get dropped and
    // become pass-by-value instead of pass-by-reference. If the functor has
    // reference arguments that it modifies, it will modify a variable on the
    // call stack rather than the original variable.
    ParameterPackDoInvokeContImpl<
        Function,
        void(_dax_pp_T___ _dax_pp_comma TransformedParameterType),
        Transform,
        RemainingArgumentsType>()(
          f,
          _dax_pp_args___(arguments) _dax_pp_comma
          transform(Access::GetFirstArgument(params)),
          transform,
          Access::GetCdr(params));
  }
};

#endif //_dax_pp_sizeof___T < DAX_MAX_PARAMETER_SIZE

template<typename Function,
         _dax_pp_typename___T _dax_pp_comma
         typename Transform>
struct ParameterPackDoInvokeContImpl<
    Function,
    void(_dax_pp_T___),
    Transform,
    ParameterPackImplNull>
{
  DAX_CONT_EXPORT
  void operator()(Function &f,
                  _dax_pp_params___(arguments) _dax_pp_comma
                  const Transform &daxNotUsed(transform),
                  ParameterPackImplNull &daxNotUsed(params))
  {
    ParameterPackCallFunctionCont<Function _dax_pp_comma _dax_pp_T___>(
          f _dax_pp_comma _dax_pp_args___(arguments));
  }
};

template<typename Function,
         _dax_pp_typename___T _dax_pp_comma
         typename Transform>
struct ParameterPackDoInvokeContImpl<
    Function,
    void(_dax_pp_T___),
    Transform,
    const ParameterPackImplNull>
{
  DAX_CONT_EXPORT
  void operator()(Function &f,
                  _dax_pp_params___(arguments) _dax_pp_comma
                  const Transform &daxNotUsed(transform),
                  const ParameterPackImplNull &daxNotUsed(params))
  {
    ParameterPackCallFunctionCont<Function _dax_pp_comma _dax_pp_T___>(
          f _dax_pp_comma _dax_pp_args___(arguments));
  }
};

// This overloaded function is here only to make errors and warnings more
// legible, particularly when the arguments given do not match the parameters
// of the functor.
template<typename Function _dax_pp_comma _dax_pp_typename___T>
DAX_EXEC_EXPORT
void ParameterPackCallFunctionExec(Function &f _dax_pp_comma
                                   _dax_pp_params___(arguments))
{
  // If you get a compiler error on the following line, it means we tried to
  // invoke a functor or worklet with the wrong arguments. Check the type of
  // FunctionType and look at its operator(). If the error is there is no
  // matched method in FunctionType::operator(), it probably means an invalid
  // argument is being passed. This is likely caused from bad arguments to
  // Dispatcher::Invoke.
  f(_dax_pp_args___(arguments));
}

#if _dax_pp_sizeof___T < DAX_MAX_PARAMETER_SIZE

template<typename Function,
         _dax_pp_typename___T _dax_pp_comma
         typename Transform,
         typename ParameterPackImplType>
struct ParameterPackDoInvokeExecImpl<
    Function,
    void(_dax_pp_T___),
    Transform,
    ParameterPackImplType>
{
  DAX_EXEC_EXPORT
  void operator()(Function &f,
                  _dax_pp_params___(arguments) _dax_pp_comma
                  const Transform &transform,
                  ParameterPackImplType &params)
  {
    typedef ParameterPackImplAccess<ParameterPackImplType> Access;
    typedef typename boost::mpl::if_<
        typename boost::is_const<ParameterPackImplType>::type,
        const typename Access::CdrType,
        typename Access::CdrType>::type RemainingArgumentsType;
    typedef typename boost::mpl::if_<
        typename boost::is_const<ParameterPackImplType>::type,
        const typename boost::remove_reference<typename Access::CarType>::type &,
        typename boost::remove_reference<typename Access::CarType>::type &>::type
          InputParameterType;
    typedef typename Transform::template ReturnType<InputParameterType>::type
        TransformedParameterType;
    // It is important to explicitly specify the template parameters. When
    // automatically determining the types, then reference types get dropped and
    // become pass-by-value instead of pass-by-reference. If the functor has
    // reference arguments that it modifies, it will modify a variable on the
    // call stack rather than the original variable.
    ParameterPackDoInvokeExecImpl<
        Function,
        void(_dax_pp_T___ _dax_pp_comma TransformedParameterType),
        Transform,
        RemainingArgumentsType>()(
          f,
          _dax_pp_args___(arguments) _dax_pp_comma
          transform(Access::GetFirstArgument(params)),
          transform,
          Access::GetCdr(params));
  }
};

#endif //_dax_pp_sizeof___T < DAX_MAX_PARAMETER_SIZE

template<typename Function,
         _dax_pp_typename___T _dax_pp_comma
         typename Transform>
struct ParameterPackDoInvokeExecImpl<
    Function,
    void(_dax_pp_T___),
    Transform,
    ParameterPackImplNull>
{
  DAX_EXEC_EXPORT
  void operator()(Function &f,
                  _dax_pp_params___(arguments) _dax_pp_comma
                  const Transform &daxNotUsed(transform),
                  ParameterPackImplNull &daxNotUsed(params))
  {
    ParameterPackCallFunctionExec<Function _dax_pp_comma _dax_pp_T___>(
          f _dax_pp_comma _dax_pp_args___(arguments));
  }
};

template<typename Function,
         _dax_pp_typename___T _dax_pp_comma
         typename Transform>
struct ParameterPackDoInvokeExecImpl<
    Function,
    void(_dax_pp_T___),
    Transform,
    const ParameterPackImplNull>
{
  DAX_EXEC_EXPORT
  void operator()(Function &f,
                  _dax_pp_params___(arguments) _dax_pp_comma
                  const Transform &daxNotUsed(transform),
                  const ParameterPackImplNull &daxNotUsed(params))
  {
    ParameterPackCallFunctionExec<Function _dax_pp_comma _dax_pp_T___>(
          f _dax_pp_comma _dax_pp_args___(arguments));
  }
};

template<typename ReturnType,
         _dax_pp_typename___T _dax_pp_comma
         typename FunctionType>
class ParameterPackReturnFunctorCont<
    ReturnType (_dax_pp_T___), FunctionType>
    : public ParameterPackReturnFunctorContBase<ReturnType, FunctionType>
{
  typedef ParameterPackReturnFunctorContBase<ReturnType, FunctionType>
      Superclass;
public:
  DAX_CONT_EXPORT
  ParameterPackReturnFunctorCont(const FunctionType &f)
    : Superclass(f)
  {  }

  DAX_CONT_EXPORT
  void operator()(_dax_pp_params___(arguments)) {
    // If you get a compiler error on the following line, it means we tried to
    // invoke a functor or worklet with the wrong arguments or the wrong return
    // value. Check the type of FunctionType and look at its operator(). If the
    // error is there is no matched method in FunctionType::operator(), it
    // probably means an invalid argument is being passed. If the error is that
    // there is no valid conversion, then the return type is probably
    // mismatched. In either case, it is likely caused from bad arguments to
    // Dispatcher::Invoke.
    const ReturnType returnValue = this->Function(_dax_pp_args___(arguments));
    this->RecordReturnValue(returnValue);
  }
};

template<typename ReturnType,
         _dax_pp_typename___T _dax_pp_comma
         typename FunctionType>
class ParameterPackReturnFunctorExec<
    ReturnType (_dax_pp_T___), FunctionType>
    : public ParameterPackReturnFunctorExecBase<ReturnType, FunctionType>
{
  typedef ParameterPackReturnFunctorExecBase<ReturnType, FunctionType>
      Superclass;
public:
  DAX_EXEC_EXPORT
  ParameterPackReturnFunctorExec(const FunctionType &f)
    : Superclass(f)
  {  }

  DAX_EXEC_EXPORT
  void operator()(_dax_pp_params___(arguments)) {
    // If you get a compiler error on the following line, it means we tried to
    // invoke a functor or worklet and the compiler could not match the
    // parenthesis operator to the call. The following are likely causes:
    //
    // 1. If the error states that the "this" argument discards the const
    //    qualifier, then it probably means that the parenthesis operator is
    //    not declared const. It should look like
    //        void operator()(...) const
    //    If instead it looks like
    //        void operator()(...)
    //    you will get a compile error here.
    //
    // 2. If the error is that there is no matched method in
    //    FunctionType::operator() and the operator is declared const, it
    //    probably means an invalid argument is being passed. This means that
    //    the argument passed to the Invoke method gets passed to an operator
    //    argument of a different type. One common reason is for this is that
    //    the arguments are given to Invoke in the wrong order. Another common
    //    reason is that the control-side object (such as ArrayHandle) is
    //    templated on the wrong basic type (e.g. dax::Scalar instead of
    //    dax::Vector3).
    //
    // 3. If the error is that there is no valid conversion, then the return
    //    type is probably mismatched. The root cause is the same as above
    //    (a bad argument to Invoke), but it is specifically pointing to the
    //    return of operator() instead of one of its arguments.
    const ReturnType returnValue = this->Function(_dax_pp_args___(arguments));
    this->RecordReturnValue(returnValue);
  }
};

template<typename ObjectToConstruct _dax_pp_comma _dax_pp_typename___T>
struct ParameterPackConstructFunctorCont<ObjectToConstruct(_dax_pp_T___)>
{
public:
  DAX_CONT_EXPORT
  ObjectToConstruct operator()(_dax_pp_params___(arguments)) const {
    return ObjectToConstruct(_dax_pp_args___(arguments));
  }
};

template<typename ObjectToConstruct _dax_pp_comma _dax_pp_typename___T>
struct ParameterPackConstructFunctorExec<ObjectToConstruct(_dax_pp_T___)>
{
public:
  DAX_EXEC_EXPORT
  ObjectToConstruct operator()(_dax_pp_params___(arguments)) const {
    return ObjectToConstruct(_dax_pp_args___(arguments));
  }
};

#if _dax_pp_sizeof___T < DAX_MAX_PARAMETER_SIZE

template<typename NewType _dax_pp_comma _dax_pp_typename___T>
struct PPackPrepend<NewType, ParameterPack<_dax_pp_T___> >
{
  typedef ParameterPack<NewType _dax_pp_comma _dax_pp_T___> type;
};

template<typename NewType,
         int Index,
         typename FirstType _dax_pp_comma
         _dax_pp_typename___T
         >
struct PPackReplace<NewType,
                    Index,
                    ParameterPack<FirstType _dax_pp_comma _dax_pp_T___> >
{
  typedef ParameterPack<FirstType _dax_pp_comma _dax_pp_T___>
      ParameterPackInputType;
  // These check for a valid index. If you get a compile error in these lines,
  // it is probably caused by an invalid Index template parameter for one of
  // the ParameterPack::Replace. Remember that parameters are index starting at
  // 1.
  BOOST_STATIC_ASSERT(Index > 0);
  BOOST_STATIC_ASSERT(Index <= ParameterPackInputType::NUM_PARAMETERS);

  typedef typename
      PPackPrepend<
          FirstType,
          typename PPackReplace<NewType, Index-1, ParameterPack<_dax_pp_T___> >::type
      >::type type;

private:
   typedef typename FindParameterPackImpl<type>::type _implType;
   typedef typename FindParameterPackImpl<ParameterPackInputType>::type
      _implInputType;

public:
   DAX_CONT_EXPORT
   static _implType ConstructImpl(NewType replacement,
                                  const _implInputType &originals)
   {
     typedef ParameterPackImplAccess<_implInputType> Access;
     typedef PPackReplace<NewType, Index-1, ParameterPack<_dax_pp_T___> >
         PPackReplaceRemainder;
     return _implType(
           Access::GetFirstArgument(originals),
           PPackReplaceRemainder::ConstructImpl(replacement,
                                                Access::GetCdr(originals)),
           ParameterPackContTag());
   }
};

template<typename NewType,
         typename FirstType _dax_pp_comma
         _dax_pp_typename___T
         >
struct PPackReplace<NewType,
                    1,
                    ParameterPack<FirstType _dax_pp_comma _dax_pp_T___> >
{
  typedef ParameterPack<FirstType _dax_pp_comma _dax_pp_T___>
      ParameterPackInputType;

  typedef ParameterPack<NewType _dax_pp_comma _dax_pp_T___> type;
  typedef typename FindParameterPackImpl<ParameterPackInputType>::type
     _implInputType;

  typedef typename FindParameterPackImpl<type>::type _implType;

  DAX_CONT_EXPORT
  static _implType ConstructImpl(NewType replacement,
                                 const _implInputType &originals)
  {
    typedef ParameterPackImplAccess<_implInputType> Access;

    return _implType(replacement,
                     Access::GetCdr(originals),
                     ParameterPackContTag());
  }
};

#endif //_dax_pp_sizeof___T < DAX_MAX_PARAMETER_SIZE

} // namespace detail

#if _dax_pp_sizeof___T > 0

template<_dax_pp_typename___T>
DAX_CONT_EXPORT
dax::internal::ParameterPack<_dax_pp_T___>
make_ParameterPack(_dax_pp_params___(arguments))
{
  typedef dax::internal::ParameterPack<_dax_pp_T___> ParameterPackType;
  return ParameterPackType(
        dax::internal::detail::FindParameterPackImpl<ParameterPackType>::
          Construct(_dax_pp_args___(arguments)),
        dax::internal::ParameterPackCopyTag(),
        dax::internal::ParameterPackContTag());
}

#endif //_dax_pp_sizeof___T > 0

template<_dax_pp_typename___T _dax_pp_comma typename ReturnType>
struct ParameterPackToSignature<ParameterPack<_dax_pp_T___>, ReturnType>
{
  typedef ReturnType type(_dax_pp_T___);
};

template<typename ReturnType _dax_pp_comma _dax_pp_typename___T>
struct SignatureToParameterPack<ReturnType(_dax_pp_T___)>
{
  typedef dax::internal::ParameterPack<_dax_pp_T___> type;
};


}
} // namespace dax::internal


#endif // !DAX_TEST_HEADER_BUILD || DAX_PARAMETER_PACK_BUILD
