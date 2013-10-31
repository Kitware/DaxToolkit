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
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_ERROR

#include <dax/cont/internal/ArrayHandleZip.h>

#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapterSerial.h>

#include <dax/exec/Assert.h>
#include <dax/exec/internal/WorkletBase.h>

#include <dax/cont/testing/Testing.h>

// TODO: Make a CUDA version of this test (to make sure everything works well
// in a distributed memory environment). You should be able to include this
// .cxx file and call the ArrayHandleZipFunctor with a different device
// adapter.

namespace {

static const int ARRAY_SIZE = 100;

template<class DeviceAdapter>
class ArrayHandleZipTestFunctor
{
private:
  typedef dax::cont::ArrayContainerControlTagBasic Container;

  typedef dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapter>
      FirstArrayType;
  typedef dax::cont::ArrayHandle<dax::Scalar,Container,DeviceAdapter>
      SecondArrayType;
  typedef dax::cont::internal::ArrayHandleZip<FirstArrayType,SecondArrayType>
      ZipArrayType;

  struct CheckZipFunctor : dax::exec::internal::WorkletBase
  {
    DAX_EXEC_CONT_EXPORT
    static dax::Id FirstValue(dax::Id index) {
      return index*2 + 1000;
    }

    DAX_EXEC_CONT_EXPORT
    static dax::Scalar SecondValue(dax::Id index) {
      return 0.0001*index + 1;
    }

    typename ZipArrayType::PortalConstExecution Portal;

    DAX_EXEC_EXPORT
    void operator()(dax::Id index) const {
      if (this->Portal.Get(index).first != this->FirstValue(index))
        {
        this->RaiseError("Bad value for first array.");
        }
      if (this->Portal.Get(index).second != this->SecondValue(index))
        {
        this->RaiseError("Bad value for second array.");
        }
    }
  };

  DAX_CONT_EXPORT
  void TestInput()
  {
    std::cout << "Checking zip as input." << std::endl;

    dax::Id firstBuffer[ARRAY_SIZE];
    dax::Scalar secondBuffer[ARRAY_SIZE];
    for (dax::Id index = 0; index < ARRAY_SIZE; index++)
      {
      firstBuffer[index] = CheckZipFunctor::FirstValue(index);
      secondBuffer[index] = CheckZipFunctor::SecondValue(index);
      }

    FirstArrayType firstArray = dax::cont::make_ArrayHandle(firstBuffer,
                                                            ARRAY_SIZE,
                                                            Container(),
                                                            DeviceAdapter());
    SecondArrayType secondArray = dax::cont::make_ArrayHandle(secondBuffer,
                                                              ARRAY_SIZE,
                                                              Container(),
                                                              DeviceAdapter());
    ZipArrayType zipArray = dax::cont::internal::make_ArrayHandleZip(
                              firstArray, secondArray);

    CheckZipFunctor functor;
    functor.Portal = zipArray.PrepareForInput();

    dax::cont::DeviceAdapterAlgorithm<DeviceAdapter>::Schedule(
          functor, ARRAY_SIZE);
  }

  struct SetZipFunctor : dax::exec::internal::WorkletBase
  {
    DAX_EXEC_CONT_EXPORT
    static dax::Id FirstValue(dax::Id index) {
      return index*3 + 2000;
    }

    DAX_EXEC_CONT_EXPORT
    static dax::Scalar SecondValue(dax::Id index) {
      return 0.0002*index + 3;
    }

    typename ZipArrayType::PortalExecution Portal;

    DAX_EXEC_EXPORT
    void operator()(dax::Id index) const {
      this->Portal.Set(index, dax::make_Pair(this->FirstValue(index),
                                             this->SecondValue(index)));
    }
  };

  DAX_CONT_EXPORT
  void TestOutput()
  {
    std::cout << "Checking zip as output." << std::endl;

    FirstArrayType firstArray;
    SecondArrayType secondArray;
    ZipArrayType zipArray = dax::cont::internal::make_ArrayHandleZip(
                              firstArray, secondArray);

    SetZipFunctor functor;
    functor.Portal = zipArray.PrepareForOutput(ARRAY_SIZE);

    dax::cont::DeviceAdapterAlgorithm<DeviceAdapter>::Schedule(
          functor, ARRAY_SIZE);

    DAX_TEST_ASSERT(zipArray.GetNumberOfValues() == ARRAY_SIZE,
                    "Zip array has wrong size.");
    DAX_TEST_ASSERT(firstArray.GetNumberOfValues() == ARRAY_SIZE,
                    "First array has wrong size.");
    DAX_TEST_ASSERT(secondArray.GetNumberOfValues() == ARRAY_SIZE,
                    "Second array has wrong size.");

    typename FirstArrayType::PortalConstControl firstPortal =
        firstArray.GetPortalConstControl();
    typename SecondArrayType::PortalConstControl secondPortal =
        secondArray.GetPortalConstControl();
    for (dax::Id index = 0; index < ARRAY_SIZE; index++)
      {
      dax::Id firstValue = firstPortal.Get(index);
      DAX_TEST_ASSERT(firstValue == SetZipFunctor::FirstValue(index),
                      "First value wrong");
      dax::Scalar secondValue = secondPortal.Get(index);
      DAX_TEST_ASSERT(secondValue == SetZipFunctor::SecondValue(index),
                      "Second value wrong");
      }
  }

public:
  void operator()() {
    this->TestInput();
    this->TestOutput();
  }
};

} // anonymous namespace

int UnitTestArrayHandleZip(int, char *[])
{
  return dax::cont::testing::Testing::Run(
        ArrayHandleZipTestFunctor<dax::cont::DeviceAdapterTagSerial>());
}
