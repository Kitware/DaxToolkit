/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_internal_TestingDeviceAdapter_h
#define __dax_cont_internal_TestingDeviceAdapter_h

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ErrorExecution.h>
#include <dax/cont/ErrorControlOutOfMemory.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

#include <dax/cont/worklet/CellGradient.h>
#include <dax/cont/worklet/Square.h>
#include <dax/cont/worklet/testing/CellMapError.h>
#include <dax/cont/worklet/testing/FieldMapError.h>

#include <dax/cont/internal/IteratorContainer.h>
#include <dax/cont/internal/Testing.h>
#include <dax/cont/internal/TestingGridGenerator.h>

#include <dax/internal/DataArray.h>

#include <dax/exec/internal/ErrorHandler.h>

#include <vector>

namespace dax {
namespace cont {
namespace internal {

#define ERROR_MESSAGE "Got an error."
#define ARRAY_SIZE 500
#define OFFSET 1000
#define DIM 64

template<class DeviceAdapter>
struct TestingDeviceAdapter
{
public:
  // Cuda kernels have to be public (in Cuda 4.0).

  struct ClearArrayKernel
  {
    DAX_EXEC_EXPORT void operator()(dax::internal::DataArray<dax::Id> array,
                                    dax::Id index,
                                    dax::exec::internal::ErrorHandler &)
    {
      array.SetValue(index, OFFSET);
    }
  };

  struct AddArrayKernel
  {
    DAX_EXEC_EXPORT void operator()(dax::internal::DataArray<dax::Id> array,
                                    dax::Id index,
                                    dax::exec::internal::ErrorHandler &)
    {
      array.SetValue(index, array.GetValue(index) + index);
    }
  };

  struct OneErrorKernel
  {
    DAX_EXEC_EXPORT void operator()(
        dax::Id, dax::Id index, dax::exec::internal::ErrorHandler &errorHandler)
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
        dax::Id, dax::Id, dax::exec::internal::ErrorHandler &errorHandler)
    {
      errorHandler.RaiseError(ERROR_MESSAGE);
    }
  };

  struct OffsetPlusIndexKernel
  {
    DAX_EXEC_EXPORT void operator()(dax::internal::DataArray<dax::Id> array,
                                    dax::Id index,
                                    dax::exec::internal::ErrorHandler &)
    {
      array.SetValue(index,OFFSET + index);
    }
  };

  struct MarkOddNumbersKernel
  {
    DAX_EXEC_EXPORT void operator()(dax::internal::DataArray<dax::Id> array,
                                    dax::Id index,
                                    dax::exec::internal::ErrorHandler &)
    {
      array.SetValue(index,index%2);
    }
  };

private:

  // Note: this test does not actually test to make sure the data is available
  // in the execution environment. It tests to make sure data gets to the array
  // and back, but it is possible that the data is not available in the
  // execution environment.
  static DAX_CONT_EXPORT void TestArrayContainerExecution()
  {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing ArrayContainerExecution" << std::endl;

    // Create original input array.
    dax::Scalar inputArray[ARRAY_SIZE];
    for (dax::Id index = 0; index < ARRAY_SIZE; index++)
      {
      inputArray[index] = index;
      }
    dax::cont::internal::IteratorContainer<dax::Scalar *>
        inputContainer(&inputArray[0], &inputArray[ARRAY_SIZE]);

    std::cout << "Allocating execution array" << std::endl;
    typename DeviceAdapter::template ArrayContainerExecution<dax::Scalar>
        executionContainer;
    executionContainer.Allocate(ARRAY_SIZE);

    std::cout << "Copying to execution array" << std::endl;
    executionContainer.CopyFromControlToExecution(inputContainer);

    // Make a destination different from the source.
    dax::Scalar outputArray[ARRAY_SIZE];
    dax::cont::internal::IteratorContainer<dax::Scalar*>
        outputContainer(&outputArray[0], &outputArray[ARRAY_SIZE]);

    std::cout << "Copying from execution array" << std::endl;
    executionContainer.CopyFromExecutionToControl(outputContainer);

    // Check the data.
    for (dax::Id index = 0; index < ARRAY_SIZE; index++)
      {
      //std::cout << inputArray[index] << ", " << outputArray[index] << std::endl;
      DAX_TEST_ASSERT(outputArray[index] == index, "Bad result.");
      }
  }

  static DAX_CONT_EXPORT void TestOutOfMemory()
  {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing Out of Memory" << std::endl;
    try
      {
      std::cout << "Do array allocation that should fail." << std::endl;
      typename DeviceAdapter::template ArrayContainerExecution<dax::Vector4>
          bigArray;
      bigArray.Allocate(-1);
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
    typename DeviceAdapter::template ArrayContainerExecution<dax::Id> array;
    array.Allocate(ARRAY_SIZE);

    std::cout << "Running clear." << std::endl;
    DeviceAdapter::Schedule(ClearArrayKernel(),
                            array.GetExecutionArray(),
                            ARRAY_SIZE);

    std::cout << "Running add." << std::endl;
    DeviceAdapter::Schedule(AddArrayKernel(),
                            array.GetExecutionArray(),
                            ARRAY_SIZE);

    std::cout << "Checking results." << std::endl;
    dax::Id controlArray[ARRAY_SIZE];
    dax::cont::internal::IteratorContainer<dax::Id*>
        arrayContainer(controlArray, controlArray + ARRAY_SIZE);
    array.CopyFromExecutionToControl(arrayContainer);

    for (dax::Id index = 0; index < ARRAY_SIZE; index++)
      {
      dax::Id value = controlArray[index];
      DAX_TEST_ASSERT(value == index + OFFSET,
                      "Got bad value for scheduled kernels.");
      }

    std::cout << "Testing Schedule on Subset" << std::endl;
    std::vector<dax::Id> rawsubset(4);
    rawsubset[0]=0;rawsubset[1]=10;rawsubset[2]=30;rawsubset[3]=20;
    dax::cont::ArrayHandle<dax::Id,DeviceAdapter> subset(rawsubset.begin(),
                                                         rawsubset.end());

    std::cout << "Running clear on subset." << std::endl;
    DeviceAdapter::Schedule(ClearArrayKernel(),
                            array.GetExecutionArray(),
                            subset);
    array.CopyFromExecutionToControl(arrayContainer);

    for (dax::Id index = 0; index < 4; index++)
      {
      dax::Id value = controlArray[rawsubset[index]];
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
    typedef dax::cont::ArrayHandle<dax::Id, DeviceAdapter> Handle;
    Handle array(ARRAY_SIZE);
    Handle result;

    //construct the index array

    DeviceAdapter::Schedule(MarkOddNumbersKernel(),
                             array.ReadyAsOutput(),
                             ARRAY_SIZE);

    DeviceAdapter::StreamCompact(array,result);
    DAX_TEST_ASSERT(result.GetNumberOfEntries() == array.GetNumberOfEntries()/2,
                    "result of compacation has an incorrect size");

    std::vector<dax::Id> resultHost;
    resultHost.resize(result.GetNumberOfEntries());

    //set the control array after stream compact as we didn't
    //know the size before hand
    result.SetNewControlData(resultHost.begin(), resultHost.end());

    //now pull the results from the execution env to the control env
    result.CompleteAsOutput();

    dax::Id index=0;
    for(std::vector<dax::Id>::const_iterator i = resultHost.begin();
        i != resultHost.end();
        ++i,++index)
      {
      const dax::Id value = *i;
      DAX_TEST_ASSERT(value == (index*2)+1,
                  "Incorrect value in compaction result.");
      }
  }

  static DAX_CONT_EXPORT void TestStreamCompactWithStencil()
  {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Testing Stream Compact with stencil" << std::endl;

    //test the version of compact that takes in input and a stencil
    typedef dax::cont::ArrayHandle<dax::Id, DeviceAdapter> Handle;

    Handle array(ARRAY_SIZE);
    Handle stencil(ARRAY_SIZE);
    Handle result;

    //construct the index array
    DeviceAdapter::Schedule(OffsetPlusIndexKernel(),
                            array.ReadyAsOutput(),
                            ARRAY_SIZE);
    DeviceAdapter::Schedule(MarkOddNumbersKernel(),
                            stencil.ReadyAsOutput(),
                            ARRAY_SIZE);

    DeviceAdapter::StreamCompact(array,stencil,result);
    DAX_TEST_ASSERT(result.GetNumberOfEntries() == array.GetNumberOfEntries()/2,
                    "result of compacation has an incorrect size");

    std::vector<dax::Id> resultHost;
    resultHost.resize(result.GetNumberOfEntries());
    //get the results from the execution env
    result.SetNewControlData(resultHost.begin(), resultHost.end());

    //copy the results from exec env to cont env
    result.CompleteAsOutput();

    dax::Id index=0;
    for(std::vector<dax::Id>::const_iterator i = resultHost.begin();
        i != resultHost.end();
        ++i,++index)
      {
      const dax::Id value = *i;
      DAX_TEST_ASSERT(value == (OFFSET + (index*2)+1),
                  "Incorrect value in compaction result.");
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
      DeviceAdapter::Schedule(OneErrorKernel(), 0, ARRAY_SIZE);
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
      DeviceAdapter::Schedule(AllErrorKernel(), 0, ARRAY_SIZE);
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
    dax::cont::internal::TestGrid<GridType,DeviceAdapter> grid(DIM);

    dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);

    std::vector<dax::Scalar> field(grid->GetNumberOfPoints());
    std::cout << "Number of Points in the grid" <<  grid->GetNumberOfPoints() << std::endl;
    for (dax::Id pointIndex = 0;
         pointIndex < grid->GetNumberOfPoints();
         pointIndex++)
      {
      field[pointIndex]
          = dax::dot(grid->GetPointCoordinates(pointIndex), trueGradient);
      }
    dax::cont::ArrayHandle<dax::Scalar, DeviceAdapter>
        fieldHandle(field.begin(), field.end());

    std::vector<dax::Scalar> square(grid->GetNumberOfPoints());
    dax::cont::ArrayHandle<dax::Scalar, DeviceAdapter>
        squareHandle(square.begin(), square.end());

    std::cout << "Running Square worklet" << std::endl;
    dax::cont::worklet::Square(grid.GetRealGrid(), fieldHandle, squareHandle);

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

    dax::cont::internal::TestGrid<GridType,DeviceAdapter> grid(DIM);

    std::cout << "Running field map worklet that errors" << std::endl;
    bool gotError = false;
    try
      {
      dax::cont::worklet::testing::FieldMapError
          <GridType, DeviceAdapter>(grid.GetRealGrid());
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

    dax::cont::internal::TestGrid<GridType,DeviceAdapter> grid(DIM);

    dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);

    std::vector<dax::Scalar> field(grid->GetNumberOfPoints());
    for (dax::Id pointIndex = 0;
         pointIndex < grid->GetNumberOfPoints();
         pointIndex++)
      {
      field[pointIndex]
          = dax::dot(grid->GetPointCoordinates(pointIndex), trueGradient);
      }
    dax::cont::ArrayHandle<dax::Scalar, DeviceAdapter>
        fieldHandle(field.begin(), field.end());

    std::vector<dax::Vector3> gradient(grid->GetNumberOfCells());
    dax::cont::ArrayHandle<dax::Vector3, DeviceAdapter>
        gradientHandle(gradient.begin(), gradient.end());

    std::cout << "Running CellGradient worklet" << std::endl;
    dax::cont::worklet::CellGradient(grid.GetRealGrid(),
                                     grid->GetPoints(),
                                     fieldHandle,
                                     gradientHandle);

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

    dax::cont::internal::TestGrid<GridType,DeviceAdapter> grid(DIM);

    std::cout << "Running cell map worklet that errors" << std::endl;
    bool gotError = false;
    try
      {
      dax::cont::worklet::testing::CellMapError
          <GridType, DeviceAdapter>(grid.GetRealGrid());
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
      TestArrayContainerExecution();
      TestOutOfMemory();
      TestSchedule();
      TestStreamCompact();
      TestStreamCompactWithStencil();
      TestErrorExecution();

      std::cout << "Doing Worklet tests with UniformGrid" << std::endl;
      WorkletTests<dax::cont::UniformGrid>();

      std::cout << "Doing Worklet tests with UnstructuredGrid" << std::endl;
      WorkletTests<dax::cont::UnstructuredGrid<dax::exec::CellHexahedron> >();
    }
  };

public:
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
