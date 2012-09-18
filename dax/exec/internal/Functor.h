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

# ifndef __dax_exec_internal_Functor_h
# define __dax_exec_internal_Functor_h
# if defined(DAX_DOXYGEN_ONLY)

# include <dax/Types.h>

/// \headerfile Functor.h dax/exec/internal/Functor.h
/// \brief Worklet invocation functor for execution environment
///
/// \tparam Invocation   Control environment invocation signature
template <typename Invocation>
class Functor
{
public:
  typedef typename dax::internal::GetNthType<0, Invocation>::type WorkletType;
  typedef dax::cont::internal::Bindings<Invocation> BindingsType;
  Functor(WorkletType worklet, BindingsType& args);
  void operator()(dax::Id id);
};

# else // !defined(DAX_DOXYGEN_ONLY)

# if !(__cplusplus >= 201103L)
#  include <dax/internal/ParameterPackCxx03.h>
# endif // !(__cplusplus >= 201103L)

# include <dax/Types.h>
# include <dax/cont/internal/Bindings.h>
# include <dax/exec/arg/Bind.h>
# include <dax/exec/internal/WorkletBase.h>
# include <dax/internal/GetNthType.h>
# include <dax/internal/Members.h>

namespace dax { namespace exec { namespace internal {

namespace detail {

struct SaveOutArgs
{
protected:
  const dax::Id Index;
  dax::exec::internal::WorkletBase& Work;
public:
  DAX_EXEC_EXPORT SaveOutArgs(dax::Id index, dax::exec::internal::WorkletBase& w):
    Index(index), Work(w)
    {}

  template <typename BindType>
  DAX_EXEC_EXPORT void operator()(BindType& execArg) const
    {
    execArg.SaveExecutionResult(Index,Work);
    }
};

template <typename Invocation>
struct FunctorMemberMap
{
  typedef typename dax::internal::GetNthType<0, Invocation>::type WorkletType;
  typedef typename WorkletType::WorkType WorkType;

  template <int Id, typename Parameter>
  struct Get
  {
    typedef typename arg::Bind<WorkType, Parameter, Invocation>::type type;
  };
};

# if __cplusplus >= 201103L
template <typename Invocation, typename ExecutionSignature, typename NumList> class FunctorImpl;
template <typename ExecutionSignature> struct FunctorNums;
template <typename T0, typename...T> struct FunctorNums<T0(T...)>
{
  typedef typename dax::internal::detail::Nums<sizeof...(T)>::type type;
};
template <typename Invocation> struct FunctorImplLookup
{
  typedef typename dax::internal::GetNthType<0, Invocation>::type WorkletType;
  typedef typename WorkletType::ExecutionSignature Sig;
  typedef FunctorImpl<Invocation, Sig, typename FunctorNums<Sig>::type> type;
};
# else // !(__cplusplus >= 201103L)
template <typename Invocation, typename ExecutionSignature> class FunctorImpl;
template <typename Invocation> struct FunctorImplLookup
{
  typedef typename dax::internal::GetNthType<0, Invocation>::type WorkletType;
  typedef FunctorImpl<Invocation, typename WorkletType::ExecutionSignature> type;
};
# endif // !(__cplusplus >= 201103L)

#define _dax_FunctorImpl_Argument(n) instance.template Get<n>()(id,this->Worklet)
#define _dax_FunctorImpl_T0          instance.template Get<0>()(id,this->Worklet) =
#define _dax_FunctorImpl_void
#define _dax_FunctorImpl(r)                                             \
public:                                                                 \
  typedef typename dax::internal::GetNthType<0, Invocation>::type       \
    WorkletType;                                                        \
  typedef dax::cont::internal::Bindings<Invocation> BindingsType;       \
private:                                                                \
  typedef dax::internal::Members<                                       \
      ExecutionSignature, FunctorMemberMap<Invocation>                  \
    > ArgumentsType;                                                    \
  WorkletType Worklet;                                                  \
  ArgumentsType Arguments;                                              \
public:                                                                 \
  FunctorImpl(WorkletType worklet, BindingsType& bindings):             \
    Worklet(worklet), Arguments(bindings) {}                            \
  DAX_EXEC_EXPORT void operator()(dax::Id id)                           \
    {                                                                   \
    ArgumentsType instance(this->Arguments);                            \
    _dax_FunctorImpl_##r                                                \
    this->Worklet(_dax_pp_enum___(_dax_FunctorImpl_Argument));          \
    instance.ForEach(SaveOutArgs(id,this->Worklet));                    \
    }

# if __cplusplus >= 201103L
#  define _dax_pp_enum___(x) x(N)...
template <typename Invocation, typename T0, typename...T, int...N>
class FunctorImpl<Invocation, T0(T...), dax::internal::detail::NumList<N...> >
{
  typedef T0 ExecutionSignature(T...);
  _dax_FunctorImpl(T0)
};
template <typename Invocation, typename...T, int...N>
class FunctorImpl<Invocation, void(T...), dax::internal::detail::NumList<N...> >
{
  typedef void ExecutionSignature(T...);
  _dax_FunctorImpl(void)
};
#  undef _dax_pp_enum___
# else // !(__cplusplus >= 201103L)
#  define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 10, <dax/exec/internal/Functor.h>))
#  include BOOST_PP_ITERATE()
# endif // !(__cplusplus >= 201103L)

# undef _dax_FunctorImpl
# undef _dax_FunctorImpl_T0
# undef _dax_FunctorImpl_void
# undef _dax_FunctorImpl_Argument

} // namespace detail

//----------------------------------------------------------------------------

template <typename Invocation>
class Functor: public detail::FunctorImplLookup<Invocation>::type
{
  typedef typename detail::FunctorImplLookup<Invocation>::type derived;
public:
  typedef typename derived::WorkletType WorkletType;
  typedef typename derived::BindingsType BindingsType;
  Functor(WorkletType worklet, BindingsType& args): derived(worklet, args) {}
};

}}} // namespace dax::exec::internal

# endif // !defined(DAX_DOXYGEN_ONLY)
# endif //__dax_exec_internal_Functor_h

#else // defined(BOOST_PP_IS_ITERATING)

template <typename Invocation, typename T0 _dax_pp_comma _dax_pp_typename___T>
class FunctorImpl<Invocation, T0(_dax_pp_T___)>
{
  typedef T0 ExecutionSignature(_dax_pp_T___);
  _dax_FunctorImpl(T0)
};

# if _dax_pp_sizeof___T > 0
template <typename Invocation, _dax_pp_typename___T>
class FunctorImpl<Invocation, void(_dax_pp_T___)>
{
  typedef void ExecutionSignature(_dax_pp_T___);
  _dax_FunctorImpl(void)
};
# endif // _dax_pp_sizeof___T > 0

#endif // defined(BOOST_PP_IS_ITERATING)
