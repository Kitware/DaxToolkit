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
#include <dax/cont/ArrayHandleCounting.h>
#include <dax/cont/ArrayHandlePermutation.h>
#include <dax/cont/DispatcherMapField.h>
#include <dax/VectorTraits.h>

#include <dax/exec/WorkletMapField.h>

#include <dax/cuda/cont/internal/testing/Testing.h>

namespace ut_Permutation
{

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

template<typename T>
struct CountByThree
{
  DAX_EXEC_CONT_EXPORT
  CountByThree(): Value() {}

  DAX_EXEC_CONT_EXPORT
  explicit CountByThree(T t): Value(t) {}

  DAX_EXEC_CONT_EXPORT
  CountByThree<T> operator+(dax::Id count) const
  { return CountByThree<T>(Value+(count*3)); }

  DAX_CONT_EXPORT
  bool operator==(const CountByThree<T>& other) const
  { return Value == other.Value; }

  T Value;
};

}


namespace dax {

/// Implement Vector Traits for CountByThree so we can use it in the execution
/// environment
template<typename T>
struct VectorTraits< ut_Permutation::CountByThree<T> >
{
  typedef ut_Permutation::CountByThree<T> ComponentType;
  static const int NUM_COMPONENTS = 1;
  typedef VectorTraitsTagSingleComponent HasMultipleComponents;

  DAX_EXEC_EXPORT
  static const ComponentType &GetComponent(const ComponentType &vector,
                                           int component) {
    return vector;
  }
  DAX_EXEC_EXPORT
  static ComponentType &GetComponent(ComponentType &vector, int component) {
    return vector;
  }

  DAX_EXEC_EXPORT static void SetComponent(ComponentType &vector,
                                           int component,
                                           ComponentType value) {
    vector = value;
  }

  DAX_EXEC_CONT_EXPORT
  static dax::Tuple<ComponentType,NUM_COMPONENTS>
  ToTuple(const ComponentType &vector)
  {
    return dax::Tuple<T,1>(vector);
  }
};

}


namespace ut_Permutation {

template< typename ValueType>
struct TemplatedTests
{
  typedef dax::cont::ArrayHandleCounting<ValueType> CountingArrayHandleType;

  typedef dax::cont::ArrayHandlePermutation<
            dax::cont::ArrayHandle<dax::Id>, //key type
            CountingArrayHandleType > ArrayPermHandleType;

  typedef dax::cont::ArrayHandlePermutation<
            dax::cont::ArrayHandleCounting<dax::Id>, //key type
            CountingArrayHandleType > ArrayCountPermHandleType;

  void operator()( const ValueType startingValue )
  {
    dax::Id everyOtherBuffer[ARRAY_SIZE/2];
    dax::Id fullBuffer[ARRAY_SIZE];

    for (dax::Id index = 0; index < ARRAY_SIZE; index++)
      {
      everyOtherBuffer[index/2] = index; //1,3,5,7,9
      fullBuffer[index] = index;
      }

  {
  //verify the different constructors work
  CountingArrayHandleType counting(startingValue,ARRAY_SIZE);
  dax::cont::ArrayHandleCounting<dax::Id> keys(dax::Id(0),ARRAY_SIZE);

  ArrayCountPermHandleType permutation_constructor(keys,counting);

  ArrayCountPermHandleType make_permutation_constructor =
      dax::cont::make_ArrayHandlePermutation(
              dax::cont::make_ArrayHandleCounting(dax::Id(0),ARRAY_SIZE),
              dax::cont::make_ArrayHandleCounting(startingValue, ARRAY_SIZE));

  }

  //make a short permutation array, verify its length and values
  {
  dax::cont::ArrayHandle<dax::Id> keys =
      dax::cont::make_ArrayHandle(everyOtherBuffer, ARRAY_SIZE/2);
  CountingArrayHandleType values(startingValue,ARRAY_SIZE);

  ArrayPermHandleType permutation =
      dax::cont::make_ArrayHandlePermutation(keys,values);

  //verify the results by executing a worklet to copy the results
  dax::cont::ArrayHandle< ValueType > result;
  dax::cont::DispatcherMapField< ut_Permutation::Pass >().Invoke(
                                                     permutation, result);

  typename dax::cont::ArrayHandle< ValueType >::PortalConstControl portal =
                                      result.GetPortalConstControl();
  ValueType correct_value = startingValue;
  for(int i=0; i < ARRAY_SIZE/2; ++i)
    {
     //the permutation should have every other value
    correct_value = correct_value + 1;
    ValueType v = portal.Get(i);
    DAX_TEST_ASSERT(v == correct_value, "Count By Three permutation wrong");
    correct_value = correct_value + 1;
    }
  }

  //make a long permutation array, verify its length and values
  {
  dax::cont::ArrayHandle<dax::Id> keys =
      dax::cont::make_ArrayHandle(fullBuffer, ARRAY_SIZE);
  CountingArrayHandleType values(startingValue,ARRAY_SIZE);

  ArrayPermHandleType permutation =
      dax::cont::make_ArrayHandlePermutation(keys,values);

  //verify the results by executing a worklet to copy the results
  dax::cont::ArrayHandle< ValueType > result;
  dax::cont::DispatcherMapField< ut_Permutation::Pass >().Invoke(
                                                     permutation, result);

  typename dax::cont::ArrayHandle< ValueType >::PortalConstControl portal =
                                      result.GetPortalConstControl();
  ValueType correct_value = startingValue;
  for(int i=0; i < ARRAY_SIZE; ++i)
    {
    ValueType v = portal.Get(i);
    DAX_TEST_ASSERT(v == correct_value, "Full permutation wrong");
    correct_value = correct_value + 1;
    }
  }


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

void TestArrayHandlePermutation()
{
  TestFunctor()( dax::Id(0) );
  TestFunctor()( dax::Scalar(0) );
  TestFunctor()( CountByThree<dax::Id>(12) );
  TestFunctor()( CountByThree<dax::Scalar>(1.2f) );
}



} // annonymous namespace

int UnitTestCudaArrayHandlePermutation(int, char *[])
{
  return dax::cuda::cont::internal::Testing::Run(
                                  ut_Permutation::TestArrayHandlePermutation);
}
