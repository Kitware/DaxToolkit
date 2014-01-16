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

//This sets up the ArrayHandle semantics to allocate pointers and share memory
//between control and execution.
#define DAX_ARRAY_CONTAINER_CONTROL DAX_ARRAY_CONTAINER_CONTROL_BASIC
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_SERIAL

#include <dax/cont/ArrayHandleCounting.h>
#include <dax/cont/ArrayHandleTransform.h>
#include <dax/cont/internal/ArrayPortalFromIterators.h>
#include <dax/cont/ArrayHandle.h>

#include <dax/cont/testing/Testing.h>

namespace {

const dax::Id ARRAY_SIZE = 10;

template<typename ValueType>
struct MySquare
{
  template<typename U>
  DAX_EXEC_EXPORT
  ValueType operator()(U u) const
    { return ValueType(u*u); }
};


template< typename ValueType, typename FunctorType >
struct TransformTests
{
  typedef  dax::cont::ArrayHandleTransform< ValueType,
                                            dax::cont::ArrayHandle< dax::Id >,
                                            FunctorType > TransformHandle;

  typedef  dax::cont::ArrayHandleTransform< ValueType,
                                    dax::cont::ArrayHandleCounting< dax::Id >,
                                    FunctorType > CountingTransformHandle;

  void operator()(const ValueType, FunctorType functor) const
  {

    //test the make_ArrayHandleTransform method
    //test a transform handle with a counting handle as the values
    CountingTransformHandle countingTransformed =
      dax::cont::make_ArrayHandleTransform<ValueType>(
        dax::cont::make_ArrayHandleCounting(dax::Id(0),ARRAY_SIZE),
        functor);

    {

    //verify that the execution portal works
    typedef typename CountingTransformHandle::PortalConstExecution CEPortal;
    CEPortal execPortal = countingTransformed.PrepareForInput();
    for(int i=0; i < ARRAY_SIZE; ++i)
      {
      const ValueType v = execPortal.Get(i);
      const ValueType correct_value = MySquare<ValueType>()(i);
      DAX_TEST_ASSERT(v == correct_value,
                      "Transform Handle with MySquare Failed");
      }
    }

    {
    //test a transform handle with a normal handle as the values
    //we are going to connect the two handles up, and than fill
    //the values and make the transform sees the new values in the handle
    dax::cont::ArrayHandle< dax::Id > input;
    TransformHandle thandle(input,functor);

    typedef dax::cont::ArrayHandle< dax::Id >::PortalExecution Portal;
    Portal execPortal = input.PrepareForOutput(ARRAY_SIZE);
    for(int i=0; i < ARRAY_SIZE; ++i)
      { execPortal.Set(i, dax::Id(i+2) ); }

    typedef typename TransformHandle::PortalConstExecution CEPortal;


    //verify that the execution portal works
    CEPortal transform_execPortal = thandle.PrepareForInput();
    for(int i=0; i < ARRAY_SIZE; ++i)
      {
      const ValueType v = transform_execPortal.Get(i);
      const ValueType correct_value = MySquare<ValueType>()(i+2);
      DAX_TEST_ASSERT(v == correct_value,
                      "Transform Handle with MySquare Failed");
      }

    //now update the array handle values again, so that
    //we can verify the transform handle gets these updated values
    for(int i=0; i < ARRAY_SIZE; ++i)
      { execPortal.Set(i, dax::Id(i*i) ); }

    //verify that the execution portal works
    transform_execPortal = thandle.PrepareForInput();
    for(int i=0; i < ARRAY_SIZE; ++i)
      {
      const ValueType v = transform_execPortal.Get(i);
      const ValueType correct_value = MySquare<ValueType>()(i*i);
      DAX_TEST_ASSERT(v == correct_value,
                      "Transform Handle with MySquare Failed");
      }
    }

  }

};


template <typename T, typename F>
void RunTransformTests(const T t, F f)
{
  TransformTests<T,F> tests;
  tests(t,f);
}

void TestArrayHandleTransform()
{
  RunTransformTests( dax::Id(0), MySquare<dax::Id>() );
  RunTransformTests( dax::Scalar(0), MySquare<dax::Scalar>() );
  RunTransformTests( dax::Vector3(dax::Scalar(0)), MySquare<dax::Vector3>() );
}



} // annonymous namespace

int UnitTestArrayHandleTransform(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestArrayHandleTransform);
}
