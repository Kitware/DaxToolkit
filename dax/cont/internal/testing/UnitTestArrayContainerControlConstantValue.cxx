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

#define DAX_ARRAY_CONTAINER_CONTROL DAX_ARRAY_CONTAINER_CONTROL_ERROR

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayPortal.h>
#include <dax/cont/ErrorControlBadValue.h>
#include <dax/cont/internal/ArrayContainerControlConstantValue.h>
#include <dax/cont/testing/Testing.h>

namespace {

const dax::Id ARRAY_SIZE = 10;

class ValueClass
 {
public:
  int v;
  ValueClass():v(0){};

  ValueClass(int i):v(i){};

  bool operator==( ValueClass other)
  {
    return (this->v == other.v);
  }
};

template< typename ValueType>
struct TemplatedTests
{
  typedef dax::cont::internal::ArrayContainerControlTagConstantValue ContainerTagType;
  typedef dax::cont::internal::ArrayContainerControl<ValueType,ContainerTagType>
                  ArrayContainerType;
  typedef dax::cont::internal::ArrayPortalConstantValue<ValueType> PortalType;


  void TestAccess( ValueType constantValue ) const
  {
  PortalType portal(constantValue,ARRAY_SIZE);

  DAX_TEST_ASSERT(portal.GetNumberOfValues() == ARRAY_SIZE,
                  "portal has wrong size.");

  typedef typename PortalType::IteratorType Iterator;
  dax::Id count = 0;
  for (Iterator i = portal.GetIteratorBegin();
       i != portal.GetIteratorEnd();
       ++i)
    {
    DAX_TEST_ASSERT( ValueType(*i) == constantValue,
                    "portal iteration has unexpected value.");
    count++;
    }

  DAX_TEST_ASSERT(count == ARRAY_SIZE, "portal iteration did go long enough.");
  }

  void TestAllocation() const
  {
    ArrayContainerType arrayContainer;

    try{ arrayContainer.GetNumberOfValues();
      DAX_TEST_ASSERT(false == true,
                      "Constant Value Container GetNumberOfValues method didn't throw error.");
      }
    catch(dax::cont::ErrorControlBadValue e){}

    try{ arrayContainer.Allocate(ARRAY_SIZE);
      DAX_TEST_ASSERT(false == true,
                    "Constant Value Container Allocate method didn't throw error.");
      }
    catch(dax::cont::ErrorControlBadValue e){}

    try
      {
      arrayContainer.Shrink(ARRAY_SIZE);
      DAX_TEST_ASSERT(true==false,
                      "Constant Value shrink do a larger size was possible. This can't be allowed.");
      }
    catch(dax::cont::ErrorControlBadValue){}

    try
      {
      arrayContainer.ReleaseResources();
      DAX_TEST_ASSERT(true==false,
                      "Can't Release an Constant Value array");
      }
    catch(dax::cont::ErrorControlBadValue){}
  }

  void operator()(const ValueType t)
  {
    TestAccess(t);
    TestAllocation();
  }
};

struct TestFunctor
{
  template <typename T>
  void operator()(const T t)
  {
    TemplatedTests<T> tests;
    tests(t);
  }
};

void TestArrayContainerConstantValue()
{
  dax::testing::Testing::TryAllTypes(TestFunctor());

  const unsigned char uc = 'a';
  const char c = 'b';
  ValueClass vc(9);

  TestFunctor()( uc );
  TestFunctor()( c );
  TestFunctor()( vc );
}

} // annonymous namespace

int UnitTestArrayContainerControlConstantValue(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestArrayContainerConstantValue);
}
