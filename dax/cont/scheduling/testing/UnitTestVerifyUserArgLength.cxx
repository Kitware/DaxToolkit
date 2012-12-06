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
#define DAX_ARRAY_CONTAINER_CONTROL DAX_ARRAY_CONTAINER_CONTROL_BASIC
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_SERIAL

#include <dax/cont/DeviceAdapter.h>

#include <dax/cont/arg/Field.h>
#include <dax/cont/internal/testing/Testing.h>
#include <dax/cont/scheduling/VerifyUserArgLength.h>
#include <dax/cont/sig/Tag.h>
#include <dax/exec/WorkletMapField.h>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/not.hpp>

namespace{

using dax::cont::arg::Field;
using dax::cont::scheduling::VerifyUserArgLength;

struct Worklet : public dax::exec::WorkletMapField
{
  typedef void ControlSignature(Field,Field);
};

void VerifyArgLength()
{

  //VerifyUserArgLength is a compile time class
  //that makes sure that we have nice error messages when a user
  //doesn't give enough, or too many arguments to a worklet when
  //it is being scheduled.

  //that means that this test is really done at compile time.


  //the nice thing about this class is that it doesn't fail to compile
  //when the arguments are wrong, it just has typedefs that are true or
  //false which than allows the user to throw the errors ( this also makes
  //the errors show up in a better contextual place )


  //verify that we detect zero arguments and not enough arguments,
  //since if the number is less than the control sig NotEnoughParameters
  //is set to false, that way it throws a compile time assert when
  //checked with DAX_ASSERT_ARG_LENGTH or BOOST_MPL_ASSERT
  typedef VerifyUserArgLength<Worklet,0>::NotEnoughParameters NEPZero;
  DAX_ASSERT_ARG_LENGTH((boost::mpl::not_<NEPZero>));

  typedef VerifyUserArgLength<Worklet,1>::NotEnoughParameters NEPOne;
  DAX_ASSERT_ARG_LENGTH((boost::mpl::not_<NEPOne>));

  //zeroArgs and oneArg should cause the TooManyParameters to be set
  //to true, since that means that in this case we don't have too many user
  //args
  typedef VerifyUserArgLength<Worklet,0>::TooManyParameters TMPZero;
  typedef VerifyUserArgLength<Worklet,1>::TooManyParameters TMPOne;
  DAX_ASSERT_ARG_LENGTH((TMPZero));
  DAX_ASSERT_ARG_LENGTH((TMPOne));


  //now lets verify that TooManyParameters work when we do have too many
  //args
  typedef VerifyUserArgLength<Worklet,3>::TooManyParameters TMPThree;
  typedef VerifyUserArgLength<Worklet,500>::TooManyParameters TMPFiveHundred;
  DAX_ASSERT_ARG_LENGTH((boost::mpl::not_<TMPThree>));
  DAX_ASSERT_ARG_LENGTH((boost::mpl::not_<TMPFiveHundred>));


  //now lets verify that Both are true when given the right length
  typedef VerifyUserArgLength<Worklet,2>::NotEnoughParameters NEPTwo;
  typedef VerifyUserArgLength<Worklet,2>::TooManyParameters TMPTwo;
  DAX_ASSERT_ARG_LENGTH((NEPTwo));
  DAX_ASSERT_ARG_LENGTH((TMPTwo));



}


}

int UnitTestVerifyUserArgLength(int, char *[])
{
  return dax::cont::internal::Testing::Run(VerifyArgLength);
}
