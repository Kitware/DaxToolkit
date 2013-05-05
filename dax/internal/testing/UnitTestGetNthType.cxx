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

#include <dax/internal/GetNthType.h>
#include <dax/testing/Testing.h>

#include <boost/type_traits/is_same.hpp>

namespace {

void expect_true(boost::true_type) {}

template <int> struct T {};

void GetNthType()
{
  typedef T<0> Sig0();
  typedef T<0> Sig1(T<1>);
  typedef T<0> Sig2(T<1>,T<2>);
  typedef T<0> Sig3(T<1>,T<2>,T<3>);
  typedef T<0> Sig4(T<1>,T<2>,T<3>,T<4>);
  typedef T<0> Sig5(T<1>,T<2>,T<3>,T<4>,T<5>);
  expect_true(boost::is_same< dax::internal::GetNthType<0,Sig0>::type , T<0> >());
  expect_true(boost::is_same< dax::internal::GetNthType<0,Sig1>::type , T<0> >());
  expect_true(boost::is_same< dax::internal::GetNthType<0,Sig2>::type , T<0> >());
  expect_true(boost::is_same< dax::internal::GetNthType<0,Sig3>::type , T<0> >());
  expect_true(boost::is_same< dax::internal::GetNthType<0,Sig4>::type , T<0> >());
  expect_true(boost::is_same< dax::internal::GetNthType<0,Sig5>::type , T<0> >());

  expect_true(boost::is_same< dax::internal::GetNthType<1,Sig1>::type , T<1> >());
  expect_true(boost::is_same< dax::internal::GetNthType<1,Sig2>::type , T<1> >());
  expect_true(boost::is_same< dax::internal::GetNthType<1,Sig3>::type , T<1> >());
  expect_true(boost::is_same< dax::internal::GetNthType<1,Sig4>::type , T<1> >());
  expect_true(boost::is_same< dax::internal::GetNthType<1,Sig5>::type , T<1> >());

  expect_true(boost::is_same< dax::internal::GetNthType<2,Sig2>::type , T<2> >());
  expect_true(boost::is_same< dax::internal::GetNthType<2,Sig3>::type , T<2> >());
  expect_true(boost::is_same< dax::internal::GetNthType<2,Sig4>::type , T<2> >());
  expect_true(boost::is_same< dax::internal::GetNthType<2,Sig5>::type , T<2> >());

  expect_true(boost::is_same< dax::internal::GetNthType<3,Sig3>::type , T<3> >());
  expect_true(boost::is_same< dax::internal::GetNthType<3,Sig4>::type , T<3> >());
  expect_true(boost::is_same< dax::internal::GetNthType<3,Sig5>::type , T<3> >());
}

} // anonymous namespace

int UnitTestGetNthType(int, char *[])
{
  return dax::testing::Testing::Run(GetNthType);
}
