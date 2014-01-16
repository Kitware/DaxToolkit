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

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayHandleImplicit.h>
#include <dax/cont/internal/ArrayPortalFromIterators.h>

#include <dax/cont/testing/Testing.h>

namespace {

const dax::Id ARRAY_SIZE = 10;

template<typename ValueType>
struct IndexSquared
{
  DAX_EXEC_CONT_EXPORT
  ValueType operator()(dax::Id i) const
    { return ValueType(i*i); }
};


template< typename ValueType, typename FunctorType >
struct ImplicitTests
{
  typedef dax::cont::ArrayHandleImplicit<ValueType,FunctorType> ImplicitHandle;

  void operator()(const ValueType, FunctorType functor) const
  {
    ImplicitHandle implict =
            dax::cont::make_ArrayHandleImplicit<ValueType>(functor,ARRAY_SIZE);

    //verify that the control portal works
    for(int i=0; i < ARRAY_SIZE; ++i)
      {
      const ValueType v = implict.GetPortalConstControl().Get(i);
      const ValueType correct_value = IndexSquared<ValueType>()(i);
        DAX_TEST_ASSERT(v == correct_value,
                      "Implicit Handle with IndexSquared Failed");
      }

    //verify that the execution portal works
    typedef typename ImplicitHandle::PortalConstExecution CEPortal;
    CEPortal execPortal = implict.PrepareForInput();
    for(int i=0; i < ARRAY_SIZE; ++i)
      {
      const ValueType v = execPortal.Get(i);
      const ValueType correct_value = IndexSquared<ValueType>()(i);
      DAX_TEST_ASSERT(v == correct_value,
                      "Implicit Handle with IndexSquared Failed");
      }
  }

};


template <typename T, typename F>
void RunImplicitTests(const T t, F f)
{
  ImplicitTests<T,F> tests;
  tests(t,f);
}

void TestArrayHandleImplicit()
{
  RunImplicitTests( dax::Id(0), IndexSquared<dax::Id>() );
  RunImplicitTests( dax::Scalar(0), IndexSquared<dax::Scalar>() );
  RunImplicitTests( dax::Vector3(dax::Scalar(0)), IndexSquared<dax::Vector3>() );
}



} // annonymous namespace

int UnitTestArrayHandleImplicit(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestArrayHandleImplicit);
}
