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
#ifndef __dax_cont_scheduling_VerifyUserArgLegnth_h
#define __dax_cont_scheduling_VerifyUserArgLegnth_h

# include <boost/mpl/if.hpp>
# include <boost/mpl/less.hpp>
# include <boost/mpl/not.hpp>
# include <boost/mpl/assert.hpp>
# include <boost/static_assert.hpp>
# include <boost/type_traits/function_traits.hpp>
# include <boost/type_traits/is_same.hpp>

#ifndef DAX_ASSERT_ARG_LENGTH
#  define DAX_ASSERT_ARG_LENGTH BOOST_MPL_ASSERT
#endif

namespace dax { namespace cont { namespace scheduling {

template<class WorkType,int NumUserArgs>
class VerifyUserArgLength
{
  typedef typename WorkType::ControlSignature ControlSig;
  typedef boost::function_traits<ControlSig> ControlSigTraits;

  BOOST_MPL_ASSERT((boost::is_same<typename ControlSigTraits::result_type, void>));

  //we define the control signature length and the number of user parameters
  //as compile time definintions to make the boost asserts easier to read
  typedef boost::mpl::int_<ControlSigTraits::arity> ControlSigLength;
  typedef boost::mpl::int_<NumUserArgs> UserLength;
public:
  //user passed in not enough parameters
  typedef boost::mpl::not_< boost::mpl::less< UserLength, ControlSigLength > >
          NotEnoughParameters;
  //user passed in too many parameters
  typedef boost::mpl::not_< boost::mpl::less< ControlSigLength, UserLength> >
          TooManyParameters;
};

} } } //dax::cont::scheduling


#endif //__dax_cont_scheduling_VerifyUserArgLegnth_h
