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

# ifndef __dax__internal__GetNthType_h
# define __dax__internal__GetNthType_h
# if defined(DAX_DOXYGEN_ONLY)

namespace dax { namespace internal {

/// \class GetNthType    GetNthType.h dax/internal/GetNthType.h
/// \tparam N            Index of the type to get.
/// \tparam TypeSequence Type sequence from which to extract elements.
///  Currently the only supported type sequence is a function type
///  of the form \code T0(T1,T2,...)
///  \endcode where index \c 0 is the return type and greater indexes
///  are the positional parameter types.
///
/// \brief Lookup elements of a type sequence by position.
template <unsigned int N, typename TypeSequence>
struct GetNthType
{
  /// Type at index \c N of \c TypeSequence
  typedef type_of_Nth_element type;
};

}} // namespace dax::internal

# else // !defined(DAX_DOXYGEN_ONLY)

# include <dax/internal/Configure.h>

# ifndef DAX_USE_VARIADIC_TEMPLATE
#  include <dax/internal/ParameterPackCxx03.h>
# endif // !DAX_USE_VARIADIC_TEMPLATE

namespace dax { namespace internal {

// Primary template is not defined.
template <unsigned int N, typename TypeSequence> struct GetNthType;

// Specialize for function types of each arity.
template <typename T0> struct GetNthType<0, T0()> { typedef T0 type; };
# ifdef DAX_USE_VARIADIC_TEMPLATE
template <typename T0, typename T1, typename... T> struct GetNthType<0, T0(T1,T...)> { typedef T0 type; };
template <typename T0, typename T1, typename... T> struct GetNthType<1, T0(T1,T...)> { typedef T1 type; };
template <unsigned int N, typename T0, typename T1, typename... T> struct GetNthType<N, T0(T1,T...)> { typedef typename GetNthType<N-1,T0(T...)>::type type; };
# else // !DAX_USE_VARIADIC_TEMPLATE
#  define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 10, <dax/internal/GetNthType.h>))
#  include BOOST_PP_ITERATE()
# endif // !DAX_USE_VARIADIC_TEMPLATE

}} // namespace dax::internal

# endif // !defined(DAX_DOXYGEN_ONLY)
# endif //__dax__internal__GetNthType_h

#else // defined(BOOST_PP_IS_ITERATING)

template <typename T0, typename T1 _dax_pp_comma _dax_pp_typename___T> struct GetNthType<0, T0(T1 _dax_pp_comma _dax_pp_T___)> { typedef T0 type; };
template <typename T0, typename T1 _dax_pp_comma _dax_pp_typename___T> struct GetNthType<1, T0(T1 _dax_pp_comma _dax_pp_T___)> { typedef T1 type; };
template <unsigned int N, typename T0, typename T1 _dax_pp_comma _dax_pp_typename___T> struct GetNthType<N, T0(T1 _dax_pp_comma _dax_pp_T___)> { typedef typename GetNthType<N-1,T0(_dax_pp_T___)>::type type; };

#endif // defined(BOOST_PP_IS_ITERATING)
