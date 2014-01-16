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

#define BOOST_SP_DISABLE_THREADS

#include <dax/cuda/cont/DeviceAdapterCuda.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayHandleImplicit.h>
#include <dax/cont/DispatcherMapField.h>

#include <dax/exec/WorkletMapField.h>

#include <dax/cuda/cont/internal/testing/Testing.h>

namespace ut_implicit{

const dax::Id ARRAY_SIZE = 10;

struct Pass : public dax::exec::WorkletMapField
{
  typedef void ControlSignature(Field(In), Field(Out));
  typedef _2 ExecutionSignature(_1);

  template<class ValueType>
  DAX_EXEC_EXPORT
  ValueType operator()(const ValueType &inValue) const
  { return inValue; }

};

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

    dax::cont::ArrayHandle< ValueType > result;
    dax::cont::DispatcherMapField< ut_implicit::Pass >().Invoke(
                                                     implict, result);

    //verify that the control portal works
    for(int i=0; i < ARRAY_SIZE; ++i)
      {
      const ValueType v = result.GetPortalConstControl().Get(i);
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



} // ut_implicit namespace

int UnitTestCudaArrayHandleImplicit(int, char *[])
{
  return dax::cuda::cont::internal::Testing::Run(
                                      ut_implicit::TestArrayHandleImplicit);
}
