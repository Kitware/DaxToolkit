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

#include <dax/cont/ArrayHandleCounting.h>
#include <dax/cont/ArrayHandleTransform.h>
#include <dax/cont/DispatcherMapField.h>

#include <dax/exec/WorkletMapField.h>

#include <dax/cuda/cont/internal/testing/Testing.h>

namespace ut_transform {

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
struct MySquare
{
  DAX_EXEC_CONT_EXPORT
  MySquare() {}

  template<typename U>
  DAX_EXEC_CONT_EXPORT
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
    dax::cont::ArrayHandleCounting< dax::Id > counting =
            dax::cont::make_ArrayHandleCounting(dax::Id(0),ARRAY_SIZE);
    CountingTransformHandle countingTransformed =
            dax::cont::make_ArrayHandleTransform<ValueType>(counting,functor);

    {
    dax::cont::ArrayHandle< ValueType > result;
    dax::cont::DispatcherMapField< ut_transform::Pass >().Invoke(
                                                     countingTransformed,
                                                     result);

    DAX_TEST_ASSERT(ARRAY_SIZE == result.GetNumberOfValues(),
                    "result handle doesn't have the correct size");
    //verify that the control portal works
    for(int i=0; i < ARRAY_SIZE; ++i)
      {
      const ValueType v = result.GetPortalConstControl().Get(i);
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

    dax::cont::DispatcherMapField< ut_transform::Pass >().Invoke(
      dax::cont::make_ArrayHandleCounting(2,ARRAY_SIZE),
      input);

    dax::cont::ArrayHandle< ValueType > result;
    dax::cont::DispatcherMapField< ut_transform::Pass >().Invoke(
                                                     thandle,
                                                     result);

    //verify that the control portal works
    for(int i=0; i < ARRAY_SIZE; ++i)
      {
      const ValueType v = result.GetPortalConstControl().Get(i);
      const ValueType correct_value = MySquare<ValueType>()(i+2);
      DAX_TEST_ASSERT(v == correct_value,
                      "Transform Handle with MySquare Failed");
      }

    //now update the array handle values again, so that
    //we can verify the transform handle gets these updated values
    dax::cont::DispatcherMapField< ut_transform::Pass >().Invoke(
      dax::cont::make_ArrayHandleCounting(500,ARRAY_SIZE),
      input);

    //verify that the transform has the new values
    dax::cont::DispatcherMapField< ut_transform::Pass >().Invoke(
                                                     thandle,
                                                     result);
    for(int i=0; i < ARRAY_SIZE; ++i)
      {
      const ValueType v = result.GetPortalConstControl().Get(i);
      const ValueType correct_value = MySquare<ValueType>()(500+i);
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
  // RunTransformTests( dax::Vector3(dax::Scalar(0)), MySquare<dax::Vector3>() );
}



} // ut_transform namespace

int UnitTestCudaArrayHandleTransform(int, char *[])
{
  return dax::cuda::cont::internal::Testing::Run(
                                     ut_transform::TestArrayHandleTransform);
}
