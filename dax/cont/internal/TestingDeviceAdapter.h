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
#ifndef __dax_cont_internal_TestingDeviceAdapter_h
#define __dax_cont_internal_TestingDeviceAdapter_h

#include <dax/cont/ArrayContainerControlBasic.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ErrorExecution.h>
#include <dax/cont/ErrorControlOutOfMemory.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

#include <dax/cont/worklet/CellGradient.h>
#include <dax/cont/worklet/Square.h>
#include <dax/cont/worklet/testing/CellMapError.h>
#include <dax/cont/worklet/testing/FieldMapError.h>

#include <dax/cont/internal/Testing.h>
#include <dax/cont/internal/TestingGridGenerator.h>
#include <dax/cont/internal/ScheduleMapAdapter.h>

#include <utility>
#include <vector>

namespace dax {
namespace cont {
namespace internal {

#define ERROR_MESSAGE "Got an error."
#define ARRAY_SIZE 500
#define OFFSET 1000
#define DIM 64

/// This class has a single static member, Run, that tests the templated
/// DeviceAdapter for conformance.
///
template<class DeviceAdapter>
struct TestingDeviceAdapter
{
private:
  typedef typename DeviceAdapter::template
      ExecutionAdapter<ArrayContainerControlBasic> ExecutionAdapter;

  typedef typename ExecutionAdapter::template FieldStructures<dax::Id>
      ::IteratorType IdIteratorType;
  typedef typename ExecutionAdapter::template FieldStructures<dax::Id>
      ::IteratorConstType IdIteratorConstType;
  typedef typename ExecutionAdapter::ErrorHandler ErrorHandler;

  typedef dax::cont
      ::ArrayHandle<dax::Id, ArrayContainerControlBasic, DeviceAdapter>
        IdArrayHandle;
  typedef typename DeviceAdapter
      ::template ArrayManagerExecution<dax::Id, ArrayContainerControlBasic>
      IdArrayManagerExecution;
  typedef ArrayContainerControlBasic<dax::Id> IdContainer;

  typedef dax::cont
      ::ArrayHandle<dax::Scalar, ArrayContainerControlBasic, DeviceAdapter>
      ScalarArrayHandle;

public:
  // Cuda kernels have to be public (in Cuda 4.0).

  struct CopyArrayKernel
  {
    DAX_EXEC_EXPORT void operator()(
        std::pair<IdIteratorConstType, IdIteratorType> arrays,
        dax::Id index,
        ErrorHandler &)
    {
      *(arrays.second + index) = *(arrays.first + index);
    }
  };

  struct ClearArrayKernel
  {
    DAX_EXEC_EXPORT void operator()(IdIteratorType array,
                                    dax::Id index,
                                    ErrorHandler &)
    {
      *(array + index) = OFFSET;
    }
  };

  struct ClearArrayMapKernel
  {
    DAX_EXEC_EXPORT void operator()(IdIteratorType array,
                                    dax::Id, dax::Id value,
                                    ErrorHandler &)
    {
      *(array + value) = OFFSET;
    }
  };

  struct AddArrayKernel
  {
    DAX_EXEC_EXPORT void operator()(IdIteratorType array,
                                    dax::Id index,
                                    ErrorHandler &)
    {
      *(array + index) += index;
    }
  };

  struct OneErrorKernel
  {
    DAX_EXEC_EXPORT void operator()(
        dax::Id, dax::Id index, ErrorHandler &errorHandler)
    {
      if (index == ARRAY_SIZE/2)
        {
        errorHandler.RaiseError(ERROR_MESSAGE);
        }
    }
  };

  struct AllErrorKernel
  {
    DAX_EXEC_EXPORT void operator()(
        dax::Id, dax::Id, ErrorHandler &errorHandler)
    {
      errorHandler.RaiseError(ERROR_MESSAGE);
    }
  };

  struct OffsetPlusIndexKernel
  {
    DAX_EXEC_EXPORT void operator()(IdIteratorType array,
                                    dax::Id index,
                                    ErrorHandler &)
    {
      *(array + index) = OFFSET + index;
    }
  };

  struct MarkOddNumbersKernel
  {
    DAX_EXEC_EXPORT void operator()(IdIteratorType array,
                                    dax::Id index,
                                    ErrorHandler &)
    {
      *(array + index) = index%2;
    }
  };

private:

  // Note: this test does not actually test to make sure the data is available
  // in the execution environment. It tests to make sure data gets to the array
  // and back, but it is possible that the data is not available in the
  // execution environment.
  static DAX_CONT_EXPORT void TestArrayManagerExecution()
  {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing ArrayManagerExecution" << std::endl;

    typedef typename DeviceAdapter
        ::template ArrayManagerExecution<dax::Scalar,ArrayContainerControlBasic>
        ArrayManagerExecution;

    // Create original input array.
    dax::Scalar inputArray[ARRAY_SIZE*2];
    for (dax::Id index = 0; index < ARRAY_SIZE*2; index++)
      {
      inputArray[index] = index;
      }
    ArrayManagerExecution inputManager;
    inputManager.LoadDataForInput(inputArray, inputArray+ARRAY_SIZE*2);

    // Change size.
    inputManager.Shrink(ARRAY_SIZE);

    // Copy array back.
    dax::Scalar outputArray[ARRAY_SIZE];
    inputManager.CopyInto(outputArray);

    // Check array.
    for (dax::Id index = 0; index < ARRAY_SIZE; index++)
      {
      DAX_TEST_ASSERT(outputArray[index] == index,
                      "Did not get correct values from array.");
      }
  }

  static DAX_CONT_EXPORT void TestOutOfMemory()
  {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing Out of Memory" << std::endl;
    try
      {
      std::cout << "Do array allocation that should fail." << std::endl;
      typename DeviceAdapter
          ::template ArrayManagerExecution<dax::Vector4, ArrayContainerControlBasic>
          bigManager;
      ArrayContainerControlBasic<dax::Vector4> supportArray;
      bigManager.AllocateArrayForOutput(supportArray, -1);
      // It does not seem reasonable to get here.  The previous call should fail.
      DAX_TEST_FAIL("A ridiculously sized allocation succeeded.  Either there "
                    "was a failure that was not reported but should have been "
                    "or the width of dax::Id is not large enough to express all "
                    "array sizes.");
      }
    catch (dax::cont::ErrorControlOutOfMemory error)
      {
      std::cout << "Got the expected error: " << error.GetMessage() << std::endl;
      }
  }

  static DAX_CONT_EXPORT void TestSchedule()
  {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing Schedule" << std::endl;

    std::cout << "Allocating execution array" << std::endl;
    IdContainer container;
    IdArrayManagerExecution manager;
    manager.AllocateArrayForOutput(container, ARRAY_SIZE);

    std::cout << "Running clear." << std::endl;
    DeviceAdapter::Schedule(ClearArrayKernel(),
                            manager.GetIteratorBegin(),
                            ARRAY_SIZE,
                            ExecutionAdapter());

    std::cout << "Running add." << std::endl;
    DeviceAdapter::Schedule(AddArrayKernel(),
                            manager.GetIteratorBegin(),
                            ARRAY_SIZE,
                            ExecutionAdapter());

    std::cout << "Checking results." << std::endl;
    manager.RetrieveOutputData(container);

    for (dax::Id index = 0; index < ARRAY_SIZE; index++)
      {
      dax::Id value = *(container.GetIteratorConstBegin() + index);
      DAX_TEST_ASSERT(value == index + OFFSET,
                      "Got bad value for scheduled kernels.");
      }

    std::cout << "Testing Schedule on Subset" << std::endl;
    const dax::Id RAWSUBSET_SIZE = 4;
    dax::Id rawsubset[RAWSUBSET_SIZE];
    rawsubset[0]=0;rawsubset[1]=10;rawsubset[2]=30;rawsubset[3]=20;
    IdArrayHandle subset(rawsubset, rawsubset + RAWSUBSET_SIZE);

    std::cout << "Running clear on subset." << std::endl;
    dax::cont::internal::ScheduleMap(ClearArrayMapKernel(),
                                     manager.GetIteratorBegin(),
                                     subset);
    manager.RetrieveOutputData(container);

    for (dax::Id index = 0; index < 4; index++)
      {
      dax::Id value = *(container.GetIteratorConstBegin() + rawsubset[index]);
      DAX_TEST_ASSERT(value == OFFSET,
                      "Got bad value for subset scheduled kernel.");
      }

  }

  static DAX_CONT_EXPORT void TestStreamCompact()
  {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing Stream Compact" << std::endl;

    //test the version of compact that takes in input and uses it as a stencil
    //and uses the index of each item as the value to place in the result vector
    IdArrayHandle array;
    IdArrayHandle result;

    //construct the index array

    DeviceAdapter::Schedule(MarkOddNumbersKernel(),
                            array.PrepareForOutput(ARRAY_SIZE).first,
                            ARRAY_SIZE,
                            ExecutionAdapter());

    DeviceAdapter::StreamCompact(array,result);
    DAX_TEST_ASSERT(result.GetNumberOfValues() == array.GetNumberOfValues()/2,
                    "result of compacation has an incorrect size");

    dax::Id index = 0;
    for (typename IdArrayHandle::IteratorConstControl iter
         = result.GetIteratorConstControlBegin();
         iter != result.GetIteratorConstControlEnd();
         iter++, index++)
      {
      const dax::Id value = *iter;
      DAX_TEST_ASSERT(value == (index*2)+1,
                      "Incorrect value in compaction results.");
      }
  }

  static DAX_CONT_EXPORT void TestStreamCompactWithStencil()
  {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing Stream Compact with stencil" << std::endl;

    IdArrayHandle array;
    IdArrayHandle stencil;
    IdArrayHandle result;

    //construct the index array
    DeviceAdapter::Schedule(OffsetPlusIndexKernel(),
                            array.PrepareForOutput(ARRAY_SIZE).first,
                            ARRAY_SIZE,
                            ExecutionAdapter());
    DeviceAdapter::Schedule(MarkOddNumbersKernel(),
                            stencil.PrepareForOutput(ARRAY_SIZE).first,
                            ARRAY_SIZE,
                            ExecutionAdapter());

    DeviceAdapter::StreamCompact(array,stencil,result);
    DAX_TEST_ASSERT(result.GetNumberOfValues() == array.GetNumberOfValues()/2,
                    "result of compacation has an incorrect size");

    dax::Id index = 0;
    for (typename IdArrayHandle::IteratorConstControl iter
         = result.GetIteratorConstControlBegin();
         iter != result.GetIteratorConstControlEnd();
         iter++, index++)
      {
      const dax::Id value = *iter;
      DAX_TEST_ASSERT(value == (OFFSET + (index*2)+1),
                  "Incorrect value in compaction result.");
      }
  }

  static DAX_CONT_EXPORT void TestOrderedUniqueValues()
  {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing Sort, Unique, and LowerBounds" << std::endl;
    dax::Id testData[ARRAY_SIZE];
    for(dax::Id i=0; i < ARRAY_SIZE; ++i)
      {
      testData[i]= OFFSET+(i % 50);
      }
    IdArrayHandle handle(testData, testData + ARRAY_SIZE);

    IdArrayHandle temp;
    DeviceAdapter::Copy(handle,temp);
    DeviceAdapter::Sort(temp);
    DeviceAdapter::Unique(temp);
    DeviceAdapter::LowerBounds(temp,handle);
    temp.ReleaseResources();

    for(dax::Id i=0; i < ARRAY_SIZE; ++i)
      {
      dax::Id value = *(handle.GetIteratorConstControlBegin() + i);
      DAX_TEST_ASSERT(value == i % 50, "Got bad value");
      }

    std::cout << "Testing Sort, Unique, and LowerBounds with random values" << std::endl;
    //now test it works when the id are not incrementing
    const dax::Id RANDOMDATA_SIZE = 6;
    dax::Id randomData[RANDOMDATA_SIZE];
    randomData[0]=500;  //2
    randomData[1]=955;  //3
    randomData[2]=955;  //3
    randomData[3]=120;  //0
    randomData[4]=320;  //1
    randomData[5]=955;  //3

    //change the control structure under the handle
    handle = IdArrayHandle(randomData, randomData + RANDOMDATA_SIZE);
    DAX_TEST_ASSERT(handle.GetNumberOfValues() == RANDOMDATA_SIZE,
                    "Handle incorrect size after setting new control data");

    DeviceAdapter::Copy(handle,temp);
    DAX_TEST_ASSERT(temp.GetNumberOfValues() == RANDOMDATA_SIZE,
                    "Copy failed");
    DeviceAdapter::Sort(temp);
    DeviceAdapter::Unique(temp);
    DeviceAdapter::LowerBounds(temp,handle);

    DAX_TEST_ASSERT(handle.GetNumberOfValues() == RANDOMDATA_SIZE,
                    "LowerBounds returned incorrect size");

    handle.CopyInto(randomData);
    DAX_TEST_ASSERT(randomData[0] == 2, "Got bad value");
    DAX_TEST_ASSERT(randomData[1] == 3, "Got bad value");
    DAX_TEST_ASSERT(randomData[2] == 3, "Got bad value");
    DAX_TEST_ASSERT(randomData[3] == 0, "Got bad value");
    DAX_TEST_ASSERT(randomData[4] == 1, "Got bad value");
    DAX_TEST_ASSERT(randomData[5] == 3, "Got bad value");
  }

  static DAX_CONT_EXPORT void TestInclusiveScan()
  {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing Inclusive Scan" << std::endl;

    //construct the index array
    IdArrayHandle array;
    DeviceAdapter::Schedule(ClearArrayKernel(),
                            array.PrepareForOutput(ARRAY_SIZE),
                            ARRAY_SIZE,
                            ExecutionAdapter());

    //we know have an array whose sum is equal to OFFSET * ARRAY_SIZE,
    //let's validate that
    dax::Id sum = DeviceAdapter::InclusiveScan(array,array);
    DAX_TEST_ASSERT(sum == OFFSET * ARRAY_SIZE, "Got bad sum from Inclusive Scan");

    //each value should be equal to the Triangle Number of that index
    //ie 1, 3, 6, 10, 15, 21 ...
    dax::Id partialSum = 0;
    dax::Id triangleNumber = 0;
    for(unsigned int i=0, pos=1; i < ARRAY_SIZE; ++i, ++pos)
      {
      const dax::Id value = *(array.GetIteratorConstControlBegin() + i);
      partialSum += value;
      triangleNumber = ((pos*(pos+1))/2);
      DAX_TEST_ASSERT(partialSum == triangleNumber * OFFSET, "Incorrect partial sum");
      }
  }

  static DAX_CONT_EXPORT void TestErrorExecution()
  {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing Exceptions in Execution Environment" << std::endl;

    std::cout << "Generating one error." << std::endl;
    std::string message;
    try
      {
      DeviceAdapter::Schedule(OneErrorKernel(),
                              0,
                              ARRAY_SIZE,
                              ExecutionAdapter());
      }
    catch (dax::cont::ErrorExecution error)
      {
      std::cout << "Got expected error: " << error.GetMessage() << std::endl;
      message = error.GetMessage();
      }
    DAX_TEST_ASSERT(message == ERROR_MESSAGE,
                    "Did not get expected error message.");

    std::cout << "Generating lots of errors." << std::endl;
    message = "";
    try
      {
      DeviceAdapter::Schedule(AllErrorKernel(),
                              0,
                              ARRAY_SIZE,
                              ExecutionAdapter());
      }
    catch (dax::cont::ErrorExecution error)
      {
      std::cout << "Got expected error: " << error.GetMessage() << std::endl;
      message = error.GetMessage();
      }
    DAX_TEST_ASSERT(message == ERROR_MESSAGE,
                    "Did not get expected error message.");
  }

  template<typename GridType>
  static DAX_CONT_EXPORT void TestWorkletMapField()
  {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing basic map field worklet" << std::endl;

    //use a scoped pointer that constructs and fills a grid of the
    //right type
    dax::cont::internal
        ::TestGrid<GridType,ArrayContainerControlBasic,DeviceAdapter> grid(DIM);

    dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);

    std::vector<dax::Scalar> field(grid->GetNumberOfPoints());
    std::cout << "Number of Points in the grid"
              <<  grid->GetNumberOfPoints()
              << std::endl;
    for (dax::Id pointIndex = 0;
         pointIndex < grid->GetNumberOfPoints();
         pointIndex++)
      {
      field[pointIndex]
          = dax::dot(grid->GetPointCoordinates(pointIndex), trueGradient);
      }
    ScalarArrayHandle fieldHandle(&field.front(), &field.back() + 1);

    ScalarArrayHandle squareHandle;

    std::cout << "Running Square worklet" << std::endl;
    dax::cont::worklet::Square(grid.GetRealGrid(), fieldHandle, squareHandle);

    std::vector<dax::Scalar> square(grid->GetNumberOfPoints());
    squareHandle.CopyInto(square);

    std::cout << "Checking result" << std::endl;
    for (dax::Id pointIndex = 0;
         pointIndex < grid->GetNumberOfPoints();
         pointIndex++)
      {
      dax::Scalar squareValue = square[pointIndex];
      dax::Scalar squareTrue = field[pointIndex]*field[pointIndex];
      DAX_TEST_ASSERT(test_equal(squareValue, squareTrue),
                      "Got bad square");
      }
  }

  template<typename GridType>
  static DAX_CONT_EXPORT void TestWorkletFieldMapError()
  {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing map field worklet error" << std::endl;

    dax::cont::internal
        ::TestGrid<GridType,ArrayContainerControlBasic,DeviceAdapter> grid(DIM);

    std::cout << "Running field map worklet that errors" << std::endl;
    bool gotError = false;
    try
      {
      dax::cont::worklet::testing::FieldMapError
          <GridType, ArrayContainerControlBasic, DeviceAdapter>
          (grid.GetRealGrid());
      }
    catch (dax::cont::ErrorExecution error)
      {
      std::cout << "Got expected ErrorExecution object." << std::endl;
      std::cout << error.GetMessage() << std::endl;
      gotError = true;
      }

    DAX_TEST_ASSERT(gotError, "Never got the error thrown.");
  }

  template<typename GridType>
  static DAX_CONT_EXPORT void TestWorkletMapCell()
  {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing basic map cell worklet" << std::endl;

    dax::cont::internal
        ::TestGrid<GridType,ArrayContainerControlBasic,DeviceAdapter> grid(DIM);

    dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);

    std::vector<dax::Scalar> field(grid->GetNumberOfPoints());
    for (dax::Id pointIndex = 0;
         pointIndex < grid->GetNumberOfPoints();
         pointIndex++)
      {
      field[pointIndex]
          = dax::dot(grid->GetPointCoordinates(pointIndex), trueGradient);
      }
    ScalarArrayHandle fieldHandle(&field.front(), &field.back() + 1);

    ScalarArrayHandle gradientHandle;

    std::cout << "Running CellGradient worklet" << std::endl;
    dax::cont::worklet::CellGradient(grid.GetRealGrid(),
                                     grid->GetPointCoordinatesArray(),
                                     fieldHandle,
                                     gradientHandle);

    std::vector<dax::Vector3> gradient(grid->GetNumberOfCells());
    gradientHandle.CopyInto(gradient.begin());

    std::cout << "Checking result" << std::endl;
    for (dax::Id cellIndex = 0;
         cellIndex < grid->GetNumberOfCells();
         cellIndex++)
      {
      dax::Vector3 gradientValue = gradient[cellIndex];
      DAX_TEST_ASSERT(test_equal(gradientValue, trueGradient),
                      "Got bad gradient");
      }
  }

  template<typename GridType>
  static DAX_CONT_EXPORT void TestWorkletCellMapError()
  {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing map cell worklet error" << std::endl;

    dax::cont::internal
        ::TestGrid<GridType,ArrayContainerControlBasic,DeviceAdapter> grid(DIM);

    std::cout << "Running cell map worklet that errors" << std::endl;
    bool gotError = false;
    try
      {
      dax::cont::worklet::testing::CellMapError
          <GridType, ArrayContainerControlBasic, DeviceAdapter>
          (grid.GetRealGrid());
      }
    catch (dax::cont::ErrorExecution error)
      {
      std::cout << "Got expected ErrorExecution object." << std::endl;
      std::cout << error.GetMessage() << std::endl;
      gotError = true;
      }

    DAX_TEST_ASSERT(gotError, "Never got the error thrown.");
  }

  struct TestAll
  {
    template<typename GridType>
    DAX_CONT_EXPORT void WorkletTests()
      {
      TestWorkletMapField<GridType>();
      TestWorkletFieldMapError<GridType>();
      TestWorkletMapCell<GridType>();
      TestWorkletCellMapError<GridType>();
      }

    DAX_CONT_EXPORT void operator()()
    {
      std::cout << "Doing DeviceAdapter tests" << std::endl;
      TestArrayManagerExecution();
      TestOutOfMemory();
      TestSchedule();
      TestStreamCompact();
      TestStreamCompactWithStencil();
      TestOrderedUniqueValues(); //tests Copy, LowerBounds, Sort, Unique
      TestInclusiveScan();
      TestErrorExecution();

      std::cout << "Doing Worklet tests with UniformGrid" << std::endl;
      WorkletTests<dax::cont::UniformGrid>();

      std::cout << "Doing Worklet tests with UnstructuredGrid" << std::endl;
      WorkletTests<dax::cont::UnstructuredGrid<dax::exec::CellHexahedron> >();
    }
  };

public:

  /// Run a suite of tests to check to see if a DeviceAdapter properly supports
  /// all members and classes required for driving Dax algorithms. Returns an
  /// error code that can be returned from the main function of a test.
  ///
  static DAX_CONT_EXPORT int Run()
  {
    return dax::cont::internal::Testing::Run(TestAll());
  }
};

#undef ERROR_MESSAGE
#undef ARRAY_SIZE
#undef OFFSET
#undef DIM

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_TestingDeviceAdapter_h
