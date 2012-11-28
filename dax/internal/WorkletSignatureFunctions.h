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

#if !BOOST_PP_IS_ITERATING
# ifndef __dax_internal_WorkletSignatureFunctions_h
# define __dax_internal_WorkletSignatureFunctions_h

# include <boost/preprocessor/iteration/iterate.hpp>
# include <boost/preprocessor/repetition/enum_shifted_params.hpp>
# include <boost/preprocessor/repetition/enum_shifted.hpp>
# include <boost/function_types/components.hpp>
# include <boost/function_types/parameter_types.hpp>

# include <boost/type_traits/is_same.hpp>

# include <boost/mpl/at.hpp>
# include <boost/mpl/contains.hpp>
# include <boost/mpl/replace.hpp>
# include <boost/mpl/size.hpp>
# include <boost/mpl/vector_c.hpp>

//Everything in this header can only be used inside the control side of dax

namespace dax { namespace internal {

namespace detail {
#   define _arg_enum___(x)      BOOST_PP_ENUM_SHIFTED(BOOST_PP_ITERATION(), _arg_enum_, x)
#   define _arg_enum_(z,n,x)    x(n)
#   define _MPL_ARG_(n) typename boost::mpl::at_c<T,n>::type

    template<int N, typename T> struct BuildSig;

#   define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 11, <dax/internal/WorkletSignatureFunctions.h>))
#   include BOOST_PP_ITERATE()

#   undef _arg_enum___
#   undef _arg_enum_
#   undef _MPL_ARG_

  template<typename Functor>
  struct ConvertToBoost
  {
    typedef boost::function_types::components<
              typename Functor::ControlSignature> ControlSignature;

    typedef boost::function_types::components<
              typename Functor::ExecutionSignature> ExecutionSignature;

    typedef boost::mpl::size<ControlSignature>   ContSize;
    typedef boost::mpl::size<ExecutionSignature> ExecSize;
  };

  template<typename Sequence, typename OldType, typename NewType>
  struct Replace
  {
    //determine if the execution arg we are searching to replace exists
    typedef typename boost::mpl::contains<Sequence,OldType>::type found;

    //We replace each element that matches the ExecArgToReplace types
    //with the ::arg::Arg<NewPlaceHolder> which we are going to next
    //push back into the control signature
    typedef typename boost::mpl::replace<
              Sequence,
              OldType,
              NewType>::type type;
  };


  template<typename Sequence, typename Type>
  struct PushBack
  {
    //push back type to the given sequence
    typedef typename boost::mpl::push_back<Sequence,Type>::type type;
  };
}

//Control side only structure
template<typename T>
struct BuildSignature
{
  typedef boost::mpl::size<T> Size;
  typedef typename dax::internal::detail::BuildSig<Size::value,T>::type type;
};


//Control side only structure
template<typename Functor, typename ExecArgToReplace, typename ExecArgToUseMetaFunc,
         typename ControlArgToUse>
struct ReplaceAndExtendSignatures
{
private:
  typedef dax::internal::detail::ConvertToBoost<Functor> BoostTypes;
  typedef typename BoostTypes::ContSize NewPlaceHolderPos;
  typedef typename ExecArgToUseMetaFunc::template apply<NewPlaceHolderPos>::type ReplacementArg;
public:

  typedef dax::internal::detail::Replace<
                      typename BoostTypes::ExecutionSignature,
                      ExecArgToReplace,
                      ReplacementArg> ReplacedExecSigArg;


  //expose our new execution signature
  typedef typename ReplacedExecSigArg::type ExecutionSignature;

  //create the struct that will return us the new control signature. We
  //always push back on the control signature, even when we didn't replace
  //anything. This makes the code easier to read, and the code that fills
  //the control signature arguments will pass a dummy argument value
  typedef typename dax::internal::detail::PushBack<
                      typename BoostTypes::ControlSignature,
                      ControlArgToUse> PushBackContSig;

  typedef typename PushBackContSig::type ControlSignature;

  //expose if we did modify the signatures
  typedef typename ReplacedExecSigArg::found found;
};

} //namespace internal
} //namespace dax

# endif //endif of __dax_internal_WorkletSignatureFunctions_h
#else

template<typename T> struct BuildSig<BOOST_PP_ITERATION(), T>
  {
  typedef typename boost::mpl::at_c<T,0>::type type(_arg_enum___(_MPL_ARG_));
  };
#endif
