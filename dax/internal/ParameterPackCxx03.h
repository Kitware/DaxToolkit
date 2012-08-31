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

# if __cplusplus >= 201103L
#  error "Do not include this header with C++11."
# else // !(__cplusplus >= 201103L)
// In C++03 use Boost.Preprocessor file iteration to approximate
// template parameter packs.
#  include <boost/preprocessor/arithmetic/dec.hpp>
#  include <boost/preprocessor/iteration/iterate.hpp>
#  include <boost/preprocessor/punctuation/comma_if.hpp>
#  include <boost/preprocessor/repetition/enum_shifted_params.hpp>
#  define _dax_pp_T___            BOOST_PP_ENUM_SHIFTED_PARAMS(BOOST_PP_ITERATION(), T___)
#  define _dax_pp_typename___T    BOOST_PP_ENUM_SHIFTED_PARAMS(BOOST_PP_ITERATION(), typename T___)
#  define _dax_pp_sizeof___T      BOOST_PP_DEC(BOOST_PP_ITERATION())
#  define _dax_pp_comma           BOOST_PP_COMMA_IF(_dax_pp_sizeof___T)
# endif // !(__cplusplus >= 201103L)

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax__internal__ParameterPackCxx03_h
