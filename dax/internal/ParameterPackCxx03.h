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
#ifndef __dax__internal__ParameterPackCxx03_h
#define __dax__internal__ParameterPackCxx03_h
#if defined(DAX_DOXYGEN_ONLY)
#else // !defined(DAX_DOXYGEN_ONLY)

#include <dax/internal/Configure.h>

#if defined(DAX_USE_VARIADIC_TEMPLATE) && !defined(DAX_TEST_HEADER_BUILD)
#  error "Do not include this header with variadic templates."
# else // !DAX_USE_VARIADIC_TEMPLATE
// In C++03 use Boost.Preprocessor file iteration to approximate
// template parameter packs.
#  include <boost/preprocessor/arithmetic/dec.hpp>
#  include <boost/preprocessor/iteration/iterate.hpp>
#  include <boost/preprocessor/punctuation/comma_if.hpp>
#  include <boost/preprocessor/repetition/enum_shifted.hpp>
#  include <boost/preprocessor/repetition/enum_shifted_binary_params.hpp>
#  include <boost/preprocessor/repetition/enum_shifted_params.hpp>
#  include <boost/preprocessor/repetition/repeat_from_to.hpp>
#  define _dax_pp_T___            BOOST_PP_ENUM_SHIFTED_PARAMS(BOOST_PP_ITERATION(), T___)
#  define _dax_pp_typename___T    BOOST_PP_ENUM_SHIFTED_PARAMS(BOOST_PP_ITERATION(), typename T___)
#  define _dax_pp_sizeof___T      BOOST_PP_DEC(BOOST_PP_ITERATION())
#  define _dax_pp_comma           BOOST_PP_COMMA_IF(_dax_pp_sizeof___T)
#  define _dax_pp_enum___(x)      BOOST_PP_ENUM_SHIFTED(BOOST_PP_ITERATION(), _dax_pp_enum, x)
#  define _dax_pp_enum(z,n,x)     _dax_pp_enum_(z,n,x)
#  define _dax_pp_enum_(z,n,x)    x(n)
#  define _dax_pp_repeat___(x)    BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_ITERATION(), _dax_pp_repeat, x)
#  define _dax_pp_repeat(z,n,x)   _dax_pp_repeat_(z,n,x)
#  define _dax_pp_repeat_(z,n,x)  x(n)
#  define _dax_pp_params___(x)    BOOST_PP_ENUM_SHIFTED_BINARY_PARAMS(BOOST_PP_ITERATION(), T___, x)
#  define _dax_pp_args___(x)      BOOST_PP_ENUM_SHIFTED_PARAMS(BOOST_PP_ITERATION(), x)
# endif // !DAX_USE_VARIADIC_TEMPLATE

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax__internal__ParameterPackCxx03_h
