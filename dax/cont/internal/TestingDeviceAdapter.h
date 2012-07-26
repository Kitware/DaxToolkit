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
template<class DeviceAdapterTag>
struct TestingDeviceAdapter
{
private:
  typedef dax::cont::ArrayContainerControlTagBasic ArrayContainerControlTag;

  typedef dax::cont
      ::ArrayHandle<dax::Id, ArrayContainerControlTag, DeviceAdapterTag>
        IdArrayHandle;
  typedef dax::cont::internal::ArrayManagerExecution<
      dax::Id, ArrayContainerControlTag, DeviceAdapterTag>
      IdArrayManagerExecution;
  typedef ArrayContainerControl<dax::Id, ArrayContainerControlTag>
      IdContainer;

  typedef typename IdArrayHandle::PortalExecution IdPortalType;
  typedef typename IdArrayHandle::PortalConstExecution IdPortalConstType;

  typedef dax::cont
      ::ArrayHandle<dax::Scalar,ArrayContainerControlTag,DeviceAdapterTag>
      ScalarArrayHandle;
  typedef dax::cont
      ::ArrayHandle<dax::Vector3,ArrayContainerControlTag,DeviceAdapterTag>
      Vector3ArrayHandle;

public:
  // Cuda kernels have to be public (in Cuda 4.0).

  struct CopyArrayKernel
  {
    DAX_CONT_EXPORT
    CopyArrayKernel(const IdPortalConstType &input,
                    const IdPortalType &output)
      : InputArray(input), OutputArray(output) {  }

    DAX_EXEC_EXPORT void operator()(
        dax::Id index,
        const dax::exec::internal::ErrorMessageBuffer &) const
    {
      this->OutputArray.Set(index, this->InputArray.Get(index));
    }

    const IdPortalConstType &InputArray;
    const IdPortalType &OutputArray;
  };

  struct ClearArrayKernel
  {
    DAX_CONT_EXPORT
    ClearArrayKernel(const IdPortalType &array) : Array(array) {  }

    DAX_EXEC_EXPORT void operator()(
        dax::Id index,
        const dax::exec::internal::ErrorMessageBuffer &) const
    {
      this->Array.Set(index, OFFSET);
    }

    const IdPortalType &Array;
  };

  struct ClearArrayMapKernel
  {
    DAX_CONT_EXPORT
    ClearArrayMapKernel(const IdPortalType &array) : Array(array) {  }

    DAX_EXEC_EXPORT void operator()(
        dax::Id,
        dax::Id value,
        const dax::exec::internal::ErrorMessageBuffer &) const
    {
      this->Array.Set(value, OFFSET);
    }

    const IdPortalType &Array;
  };

  struct AddArrayKernel
  {
    DAX_CONT_EXPORT
    AddArrayKernel(const IdPortalType &array) : Array(array) {  }

    DAX_EXEC_EXPORT void operator()(
        dax::Id index,
        const dax::exec::internal::ErrorMessageBuffer &) const
    {
      this->Array.Set(index, this->Array.Get(index) + index);
    }

    const IdPortalType &Array;
  };

  struct OneErrorKernel
  {
    DAX_EXEC_EXPORT void operator()(
        dax::Id index,
        const dax::exec::internal::ErrorMessageBuffer &errorMessage) const
    {
      if (index == ARRAY_SIZE/2)
        {
        errorMessage.RaiseError(ERROR_MESSAGE);
        }
    }
  };

  struct AllErrorKernel
  {
    DAX_EXEC_EXPORT void operator()(
        dax::Id daxNotUsed(index),
        const dax::exec::internal::ErrorMessageBuffer &errorMessage) const
    {
      errorMessage.RaiseError(ERROR_MESSAGE);
    }
  };

  struct OffsetPlusIndexKernel
  {
    DAX_CONT_EXPORT
    OffsetPlusIndexKernel(const IdPortalType &array) : Array(array) {  }

    DAX_EXEC_EXPORT void operator()(
        dax::Id index,
        const dax::exec::internal::ErrorMessageBuffer &) const
    {
      this->Array.Set(index, OFFSET + index);
    }

    const IdPortalType &Array;
  };

  struct MarkOddNumbersKernel
  {
    DAX_CONT_EXPORT
    MarkOddNumbersKernel(const IdPortalType &array) : Array(array) {  }

    DAX_EXEC_EXPORT void operator()(
        dax::Id index,
        const dax::exec::internal::ErrorMessageBuffer &) const
    {
      this->Array.Set(index, index%2);
    }

    const IdPortalType &Array;
  };

private:

  template<typename T>
  static DAX_CONT_EXPORT
  dax::cont::ArrayHandle<T, ArrayContainerControlTagBasic, DeviceAdapterTag>
  MakeArrayHandle(const T *array, dax::Id length)
  {
    return dax::cont::make_ArrayHandle(array,
                                       length,
                                       ArrayContainerControlTagBasic(),
                                       DeviceAdapterTag());
  }

  template<typename T>
  static DAX_CONT_EXPORT
  dax::cont::ArrayHandle<T, ArrayContainerControlTagBasic, DeviceAdapterTag>
  MakeArrayHandle(const std::vector<T> array)
  {
    return dax::cont::make_ArrayHandle(array,
                                       ArrayContainerControlTagBasic(),
                                       DeviceAdapterTag());
  }

  // Note: this test does not actually test to make sure the data is available
  // in the execution environment. It tests to make sure data gets to the array
  // and back, but it is possible that the data is not available in the
  // execution environment.
  static DAX_CONT_EXPORT void TestArrayManagerExecution()
  {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing ArrayManagerExecution" << std::endl;

    typedef dax::cont::internal::ArrayManagerExecution<
        dax::Scalar,ArrayContainerControlTagBasic,DeviceAdapterTag>
        ArrayManagerExecution;

    // Create original input array.
    dax::Scalar inputArray[ARRAY_SIZE*2];
    for (dax::Id index = 0; index < ARRAY_SIZE*2; index++)
      {
      inputArray[index] = index;
      }
    dax::cont::ArrayPortalFromIterators<dax::Scalar *>
        inputPortal(inputArray, inputArray+ARRAY_SIZE*2);
    ArrayManagerExecution inputManager;
    inputManager.LoadDataForInput(inputPortal);

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
    // Only test out of memory with 64 bit ids.  If there are 32 bit ids on
    // a 64 bit OS (common), it is simply too hard to get a reliable allocation
    // that is too much memory.
#ifdef DAX_USE_64BIT_IDS
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing Out of Memory" << std::endl;
    try
      {
      std::cout << "Do array allocation that should fail." << std::endl;
      dax::cont::internal::ArrayManagerExecution<
          dax::Vector4,ArrayContainerControlTagBasic,DeviceAdapterTag>
          bigManager;
      dax::cont::internal::ArrayContainerControl<
          dax::Vector4, ArrayContainerControlTagBasic> supportArray;
      const dax::Id bigSize = 0x7FFFFFFFFFFFFFFFLL;
      bigManager.AllocateArrayForOutput(supportArray, bigSize);
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
#else
    std::cout << "--------- Skiping out of memory test" << std::endl;
#endif
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
    dax::cont::internal::Schedule(ClearArrayKernel(manager.GetPortal()),
                                  ARRAY_SIZE,
                                  DeviceAdapterTag());

    std::cout << "Running add." << std::endl;
    dax::cont::internal::Schedule(AddArrayKernel(manager.GetPortal()),
                                  ARRAY_SIZE,
                                  DeviceAdapterTag());

    std::cout << "Checking results." << std::endl;
    manager.RetrieveOutputData(container);

    for (dax::Id index = 0; index < ARRAY_SIZE; index++)
      {
      dax::Id value = container.GetPortalConst().Get(index);
      DAX_TEST_ASSERT(value == index + OFFSET,
                      "Got bad value for scheduled kernels.");
      }

    std::cout << "Testing Schedule on Subset" << std::endl;
    const dax::Id RAWSUBSET_SIZE = 4;
    dax::Id rawsubset[RAWSUBSET_SIZE];
    rawsubset[0]=0;rawsubset[1]=10;rawsubset[2]=30;rawsubset[3]=20;
    IdArrayHandle subset = MakeArrayHandle(rawsubset, RAWSUBSET_SIZE);

    std::cout << "Running clear on subset." << std::endl;
    dax::cont::internal::ScheduleMap(ClearArrayMapKernel(manager.GetPortal()),
                                     subset);
    manager.RetrieveOutputData(container);

    for (dax::Id index = 0; index < 4; index++)
      {
      dax::Id value = container.GetPortalConst().Get(rawsubset[index]);
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

    dax::cont::internal::Schedule(
          MarkOddNumbersKernel(array.PrepareForOutput(ARRAY_SIZE)),
          ARRAY_SIZE,
          DeviceAdapterTag());

    dax::cont::internal::StreamCompact(array, result, DeviceAdapterTag());
    DAX_TEST_ASSERT(result.GetNumberOfValues() == array.GetNumberOfValues()/2,
                    "result of compacation has an incorrect size");

    for (dax::Id index = 0; index < result.GetNumberOfValues(); index++)
      {
      const dax::Id value = result.GetPortalConstControl().Get(index);
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
    dax::cont::internal::Schedule(
          OffsetPlusIndexKernel(array.PrepareForOutput(ARRAY_SIZE)),
          ARRAY_SIZE,
          DeviceAdapterTag());
    dax::cont::internal::Schedule(
          MarkOddNumbersKernel(stencil.PrepareForOutput(ARRAY_SIZE)),
          ARRAY_SIZE,
          DeviceAdapterTag());

    dax::cont::internal::StreamCompact(array,stencil,result,DeviceAdapterTag());
    DAX_TEST_ASSERT(result.GetNumberOfValues() == array.GetNumberOfValues()/2,
                    "result of compacation has an incorrect size");

    for (dax::Id index = 0; index < result.GetNumberOfValues(); index++)
      {
      const dax::Id value = result.GetPortalConstControl().Get(index);
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
    IdArrayHandle input = MakeArrayHandle(testData, ARRAY_SIZE);

    IdArrayHandle handle;
    IdArrayHandle temp;
    dax::cont::internal::Copy(input,handle,DeviceAdapterTag());
    dax::cont::internal::Copy(handle,temp,DeviceAdapterTag());
    dax::cont::internal::Sort(temp,DeviceAdapterTag());
    dax::cont::internal::Unique(temp,DeviceAdapterTag());
    dax::cont::internal::LowerBounds(temp,input,handle,DeviceAdapterTag());
    temp.ReleaseResources();

    for(dax::Id i=0; i < ARRAY_SIZE; ++i)
      {
      dax::Id value = handle.GetPortalConstControl().Get(i);
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
    input = MakeArrayHandle(randomData, RANDOMDATA_SIZE);
    dax::cont::internal::Copy(input,handle,DeviceAdapterTag());
    DAX_TEST_ASSERT(handle.GetNumberOfValues() == RANDOMDATA_SIZE,
                    "Handle incorrect size after setting new control data");

    dax::cont::internal::Copy(handle,temp,DeviceAdapterTag());
    DAX_TEST_ASSERT(temp.GetNumberOfValues() == RANDOMDATA_SIZE,
                    "Copy failed");
    dax::cont::internal::Sort(temp,DeviceAdapterTag());
    dax::cont::internal::Unique(temp,DeviceAdapterTag());
    dax::cont::internal::LowerBounds(temp,handle,DeviceAdapterTag());

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
    dax::cont::internal::Schedule(
          ClearArrayKernel(array.PrepareForOutput(ARRAY_SIZE)),
          ARRAY_SIZE,
          DeviceAdapterTag());

    //we know have an array whose sum is equal to OFFSET * ARRAY_SIZE,
    //let's validate that
    dax::Id sum = dax::cont::internal::InclusiveScan(array,
                                                     array,
                                                     DeviceAdapterTag());
    DAX_TEST_ASSERT(sum == OFFSET * ARRAY_SIZE,
                    "Got bad sum from Inclusive Scan");

    //each value should be equal to the Triangle Number of that index
    //ie 1, 3, 6, 10, 15, 21 ...
    dax::Id partialSum = 0;
    dax::Id triangleNumber = 0;
    for(unsigned int i=0, pos=1; i < ARRAY_SIZE; ++i, ++pos)
      {
      const dax::Id value = array.GetPortalConstControl().Get(i);
      partialSum += value;
      triangleNumber = ((pos*(pos+1))/2);
      DAX_TEST_ASSERT(partialSum == triangleNumber * OFFSET,
                      "Incorrect partial sum");
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
      dax::cont::internal::Schedule(OneErrorKernel(),
                                    ARRAY_SIZE,
                                    DeviceAdapterTag());
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
      dax::cont::internal::Schedule(AllErrorKernel(),
                                    ARRAY_SIZE,
                                    DeviceAdapterTag());
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
        ::TestGrid<GridType,ArrayContainerControlTagBasic,DeviceAdapterTag>
        grid(DIM);

    dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);

    std::vector<dax::Scalar> field(grid->GetNumberOfPoints());
    std::cout << "Number of Points in the grid: "
              <<  grid->GetNumberOfPoints()
              << std::endl;
    for (dax::Id pointIndex = 0;
         pointIndex < grid->GetNumberOfPoints();
         pointIndex++)
      {
      dax::Vector3 coordinates = grid.GetPointCoordinates(pointIndex);
      field[pointIndex] = dax::dot(coordinates, trueGradient);
      }
    ScalarArrayHandle fieldHandle = MakeArrayHandle(field);

    ScalarArrayHandle squareHandle;

    std::cout << "Running Square worklet" << std::endl;
    dax::cont::worklet::Square(fieldHandle, squareHandle);

    std::vector<dax::Scalar> square(grid->GetNumberOfPoints());
    squareHandle.CopyInto(square.begin());

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
        ::TestGrid<GridType,ArrayContainerControlTagBasic,DeviceAdapterTag>
        grid(DIM);

    std::cout << "Running field map worklet that errors" << std::endl;
    bool gotError = false;
    try
      {
      dax::cont::worklet::testing::FieldMapError(
            grid.GetRealGrid().GetPointCoordinates());
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

    if (GridType::CellType::TOPOLOGICAL_DIMENSIONS < 3)
      {
      std::cout << "Skipping.  Too hard to check gradient "
                << "on cells with topological dimension < 3" << std::endl;
      }
    else
      {
      // Calling a separate Impl function because the CUDA compiler is good
      // enough to optimize the if statement as a constant expression and
      // then complains about unreachable statements after a return.
      TestWorkletMapCellImpl<GridType>();
      }
  }

  template<typename GridType>
  static DAX_CONT_EXPORT void TestWorkletMapCellImpl()
  {
    dax::cont::internal
        ::TestGrid<GridType,ArrayContainerControlTagBasic,DeviceAdapterTag>
        grid(DIM);

    dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);

    std::vector<dax::Scalar> field(grid->GetNumberOfPoints());
    for (dax::Id pointIndex = 0;
         pointIndex < grid->GetNumberOfPoints();
         pointIndex++)
      {
      dax::Vector3 coordinates = grid.GetPointCoordinates(pointIndex);
      field[pointIndex] = dax::dot(coordinates, trueGradient);
      }
    ScalarArrayHandle fieldHandle = MakeArrayHandle(field);

    Vector3ArrayHandle gradientHandle;

    std::cout << "Running CellGradient worklet" << std::endl;
    dax::cont::worklet::CellGradient(grid.GetRealGrid(),
                                     grid->GetPointCoordinates(),
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
        ::TestGrid<GridType,ArrayContainerControlTagBasic,DeviceAdapterTag>
        grid(DIM);

    std::cout << "Running cell map worklet that errors" << std::endl;
    bool gotError = false;
    try
      {
      dax::cont::worklet::testing::CellMapError(grid.GetRealGrid(),
                                                DeviceAdapterTag());
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
      WorkletTests<dax::cont::UniformGrid<> >();

      std::cout << "Doing Worklet tests with UnstructuredGrid types" << std::endl;
      WorkletTests<dax::cont::UnstructuredGrid<dax::exec::CellHexahedron> >();
      WorkletTests<dax::cont::UnstructuredGrid<dax::exec::CellTriangle> >();
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
