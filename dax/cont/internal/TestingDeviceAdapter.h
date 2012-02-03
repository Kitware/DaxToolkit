/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_internal_TestingDeviceAdapter_h
#define __dax_cont_internal_TestingDeviceAdapter_h

#include <dax/cont/ErrorExecution.h>

#include <dax/cont/internal/IteratorContainer.h>
#include <dax/cont/internal/Testing.h>

#include <dax/internal/DataArray.h>

#include <dax/exec/internal/ErrorHandler.h>

namespace dax {
namespace cont {
namespace internal {

#define ERROR_MESSAGE "Got an error."

template<class DeviceAdapter>
struct TestingDeviceAdapter
{
private:
  static const dax::Id ARRAY_SIZE = 500;
  static const dax::Id OFFSET = 1000;

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

  struct TestAll
  {
    DAX_CONT_EXPORT void operator()()
    {
      std::cout << "Doing DeviceAdapter tests" << std::endl;
      TestArrayContainerExecution();
      TestSchedule();
      TestErrorExecution();
    }
  };

public:
  static DAX_CONT_EXPORT int Run()
  {
    return dax::cont::internal::Testing::Run(TestAll());
  }
};

#undef ERROR_MESSAGE

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_TestingDeviceAdapter_h
