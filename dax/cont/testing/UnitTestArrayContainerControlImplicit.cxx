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

#include <dax/Types.h>
#include <dax/cont/ArrayContainerControlImplicit.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/internal/IteratorFromArrayPortal.h>

#include <dax/VectorTraits.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/testing/Testing.h>

namespace
{

template <typename T>
struct TestImplicitContainer
{
  typedef T ValueType;
  ValueType Temp;
  typedef dax::cont::internal::IteratorFromArrayPortal<
      TestImplicitContainer<T> > IteratorType;


  DAX_EXEC_CONT_EXPORT
  TestImplicitContainer(): Temp(1) {  }

  DAX_EXEC_CONT_EXPORT
  dax::Id GetNumberOfValues() const
  {
    return 1;
  }

  DAX_EXEC_CONT_EXPORT
  ValueType Get(dax::Id daxNotUsed(index)) const
  {
  return Temp;
  }

  DAX_CONT_EXPORT
  IteratorType GetIteratorBegin() const {
    return IteratorType(*this);
  }

  DAX_CONT_EXPORT
  IteratorType GetIteratorEnd() const {
    return IteratorType(*this, this->GetNumberOfValues());
  }

};

const dax::Id ARRAY_SIZE = 1;


template <typename T>
struct TemplatedTests
{
  typedef dax::cont::ArrayContainerControlTagImplicit<
      TestImplicitContainer<T> > ContainerTagType;
  typedef dax::cont::internal::ArrayContainerControl<
      T,  ContainerTagType > ArrayContainerType;

  typedef typename ArrayContainerType::ValueType ValueType;
  typedef typename ArrayContainerType::PortalType PortalType;
  typedef typename PortalType::IteratorType IteratorType;

  void BasicAllocation()
  {
    ArrayContainerType arrayContainer;

    try{ arrayContainer.GetNumberOfValues();
      DAX_TEST_ASSERT(false == true,
                      "Implicit Container GetNumberOfValues method didn't throw error.");
      }
    catch(dax::cont::ErrorControlBadValue e){}

    try{ arrayContainer.Allocate(ARRAY_SIZE);
      DAX_TEST_ASSERT(false == true,
                    "Implicit Container Allocate method didn't throw error.");
      }
    catch(dax::cont::ErrorControlBadValue e){}

    try
      {
      arrayContainer.Shrink(ARRAY_SIZE);
      DAX_TEST_ASSERT(true==false,
                      "Array shrink do a larger size was possible. This can't be allowed.");
      }
    catch(dax::cont::ErrorControlBadValue){}

    try
      {
      arrayContainer.ReleaseResources();
      DAX_TEST_ASSERT(true==false,
                      "Can't Release an implicit array");
      }
    catch(dax::cont::ErrorControlBadValue){}
  }

  void BasicAccess()
    {
    TestImplicitContainer<T> portal;
    dax::cont::ArrayHandle<T,ContainerTagType> implictHandle(portal);
    DAX_TEST_ASSERT(implictHandle.GetNumberOfValues() == 1,
                    "handle should have size 1");
    DAX_TEST_ASSERT(implictHandle.GetPortalConstControl().Get(0) == T(1),
                    "portals first values should be 1");
    ;
    }

  void operator()()
  {
    BasicAllocation();
    BasicAccess();
  }
};

struct TestFunctor
{
  template <typename T>
  void operator()(T)
  {
    TemplatedTests<T> tests;
    tests();

  }
};

void TestArrayContainerControlBasic()
{
  dax::testing::Testing::TryAllTypes(TestFunctor());
}

} // Anonymous namespace

int UnitTestArrayContainerControlImplicit(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestArrayContainerControlBasic);
}
