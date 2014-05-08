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
# ifndef __dax_exec_internal_Functor_h
# define __dax_exec_internal_Functor_h

# include <dax/Types.h>
# include <dax/cont/internal/Bindings.h>
# include <dax/exec/arg/FindBinding.h>
# include <dax/exec/internal/IJKIndex.h>
# include <dax/exec/internal/WorkletBase.h>
# include <dax/internal/GetNthType.h>
# include <dax/internal/Members.h>

#include <boost/type_traits/remove_reference.hpp>
#include <boost/utility/enable_if.hpp>

namespace dax { namespace exec { namespace internal {

namespace detail {

template <typename Invocation>
struct FunctorMemberMap
{
  template <int Id, typename Parameter>
  struct Get
  {
  typedef typename arg::FindBinding<Invocation, Parameter>::type type;
  };
};

struct FunctorWorkletMemberMap
{
  template<int N, typename ExecArgType>
  struct Get {
    typedef typename ExecArgType::ReturnType type;
  };
};

template<typename ExecBindings>
struct FunctorExecBindingToWorkletBinding
{
  typedef dax::internal::Members<
      typename dax::internal::ParameterPackToSignature<
        typename ExecBindings::ParameterPackType,
        typename ExecBindings::ReturnType>::type,
      FunctorWorkletMemberMap> type;
};

template<typename IndexType>
struct FunctorGetArgs
{
protected:
  const IndexType Index;
  const dax::exec::internal::WorkletBase &Work;
public:
  DAX_EXEC_EXPORT FunctorGetArgs(IndexType index,
                                 const dax::exec::internal::WorkletBase &w)
    : Index(index), Work(w) {  }

  template<typename BindType>
  DAX_EXEC_EXPORT
  typename BindType::ReturnType
  operator()(BindType &execArg) const
  {
    return execArg(IndexGetValue(this->Index), this->Work);
  }
};

template<typename IndexType>
struct FunctorSaveArgs
{
protected:
  const IndexType Index;
  const dax::exec::internal::WorkletBase& Work;
public:
  DAX_EXEC_EXPORT FunctorSaveArgs(IndexType index,
                                  const dax::exec::internal::WorkletBase& w)
    : Index(index), Work(w) {  }

  template <typename BindType>
  DAX_EXEC_EXPORT
  void operator()(BindType& execArg) const
    {
    execArg.SaveExecutionResult(this->Index, this->Work);
    }
};

template<typename IndexType, typename WorkletType>
struct FunctorTransformExecArg
{
  template<typename ExecArgType>
  struct ReturnType {
    typedef typename boost::remove_reference<ExecArgType>::type::ReturnType type;
  };

  DAX_EXEC_EXPORT
  FunctorTransformExecArg(const IndexType &index, const WorkletType &worklet)
    : Index(index), Worklet(worklet) {  }

  template<typename ExecArgType>
  DAX_EXEC_EXPORT
  typename ExecArgType::ReturnType
  operator()(ExecArgType &argument) const
  {
    return argument(this->Index, this->Worklet);
  }

  const IndexType &Index;
  const WorkletType &Worklet;
};

} // namespace detail

//----------------------------------------------------------------------------

/// \headerfile Functor.h dax/exec/internal/Functor.h
/// \brief Worklet invocation functor for execution environment
///
/// \tparam Invocation An instance of dax::internal::Invocation that contains
/// information about the worklet and how it was instantiated.
template<typename Invocation>
class Functor
{
public:
  typedef typename Invocation::Worklet WorkletType;
  typedef typename WorkletType::ExecutionSignature ExecutionSignature;
  typedef typename dax::cont::internal::Bindings<Invocation>::type BindingsType;

  DAX_CONT_EXPORT
  Functor(WorkletType worklet, BindingsType &bindings)
    : Worklet(worklet),
      Arguments(bindings,
                dax::internal::MembersInitialArgumentTag(),
                dax::internal::MembersContTag()) {  }

  DAX_CONT_EXPORT
  void SetErrorMessageBuffer(
      const dax::exec::internal::ErrorMessageBuffer &errorBuffer)
  {
    this->Worklet.SetErrorMessageBuffer(errorBuffer);
  }

  DAX_EXEC_EXPORT
  void operator()(dax::Id index) const
  {
    this->InvokeWorklet(index);
  }
  DAX_EXEC_EXPORT
  void operator()(dax::exec::internal::IJKIndex index) const
  {
    this->InvokeWorklet(index);
  }

private:
  WorkletType Worklet;

  typedef dax::internal::Members<
      ExecutionSignature,
      detail::FunctorMemberMap<Invocation>
    > ArgumentsType;
#ifdef DAX_CUDA
  const ArgumentsType Arguments;
#else
  mutable ArgumentsType Arguments;
#endif

  template<typename IndexType>
  DAX_EXEC_EXPORT
  void InvokeWorklet(IndexType index) const
  {
    // Make a copy of the Arguments object, which should remain constant
    // for thread performance (?)
#ifdef DAX_CUDA
    ArgumentsType instance(this->Arguments);
#else
    ArgumentsType& instance(this->Arguments);
#endif

    // Invoke the worklet with these arguments.
    this->DoInvokeWorklet<ArgumentsType::FIRST_INDEX>(instance, index);

    // Save the results back to any arrays in the ExecArgs bindings. The
    // WorkletBindingsType should contain reference types back to any output
    // types in this->Arguments, so they can be directly written back.
    instance.ForEachExec(
          detail::FunctorSaveArgs<IndexType>(index, this->Worklet));
  }

  template<int FirstIndex, typename IndexType>
  DAX_EXEC_EXPORT
  typename boost::enable_if_c<FirstIndex == 0>::type
  DoInvokeWorklet(ArgumentsType &argumentsInstance,
                  const IndexType &index) const
  {
    typedef typename ArgumentsType::ReturnType::ReturnType ReturnType;
    argumentsInstance.template Get<0>()(index,this->Worklet) =
        dax::internal::ParameterPackInvokeWithReturnExec<
            typename boost::remove_reference<ReturnType>::type>(
          this->Worklet,
          argumentsInstance.GetArgumentValues(),
          detail::FunctorTransformExecArg<IndexType,WorkletType>(
            index, this->Worklet));
  }

  template<int FirstIndex, typename IndexType>
  DAX_EXEC_EXPORT
  typename boost::enable_if_c<FirstIndex != 0>::type
  DoInvokeWorklet(ArgumentsType &argumentsInstance,
                  const IndexType &index) const
  {
    dax::internal::ParameterPackInvokeExec(
          this->Worklet,
          argumentsInstance.GetArgumentValues(),
          detail::FunctorTransformExecArg<IndexType,WorkletType>(
            index, this->Worklet));
  }
};

}}} // namespace dax::exec::internal

# endif //__dax_exec_internal_Functor_h
