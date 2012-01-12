/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/internal/ArrayContainerExecution.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <dax/cont/internal/IteratorContainer.h>

#include <algorithm>

namespace {

const dax::Id ARRAY_SIZE = 10;

#define test_assert(condition, message) \
  test_assert_impl(condition, message, __FILE__, __LINE__);

static inline void test_assert_impl(bool condition,
                                    const std::string& message,
                                    const char *file,
                                    int line)
{
  if(!condition)
    {
    std::stringstream error;
    error << file << ":" << line << std::endl;
    error << message << std::endl;
    throw error.str();
    }
}

//-----------------------------------------------------------------------------
struct AddOne
{
  void operator()(dax::Scalar &x) {
    x += 1;
  }
};

//-----------------------------------------------------------------------------
static void TestBasicTransfers()
{
  // Create original input array.
  dax::Scalar inputArray[ARRAY_SIZE];
  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    inputArray[index] = index;
    }
  dax::cont::internal::IteratorContainer<dax::Scalar*>
      inputContainer(&inputArray[0], &inputArray[ARRAY_SIZE]);

  // Create the managed container for the array in the execution environment.
  dax::cont::internal::ArrayContainerExecution<dax::Scalar> executionContainer;
  executionContainer.Allocate(ARRAY_SIZE);

  // Copy the data to the execution environment.
  executionContainer.CopyFromControlToExecution(inputContainer);

  // Do something with the array on the device.
  dax::internal::DataArray<dax::Scalar> executionArray
      = executionContainer.GetExecutionArray();
  std::for_each(executionArray.GetPointer(),
                executionArray.GetPointer() + ARRAY_SIZE,
                AddOne());

  // Make a destination to check results.
  dax::Scalar outputArray[ARRAY_SIZE];
  dax::cont::internal::IteratorContainer<dax::Scalar*>
      outputContainer(&outputArray[0], &outputArray[ARRAY_SIZE]);

  // Copy the results data back from the execution environment.
  executionContainer.CopyFromExecutionToControl(outputContainer);

  // Check the results.
  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    std::cout << inputArray[index] << ", " << outputArray[index] << std::endl;
    test_assert(outputArray[index] == index+1, "Bad result.");
    }
}

} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestArrayContainerExecution(int, char *[])
{
  try
    {
    TestBasicTransfers();
    }
  catch (std::string error)
    {
    std::cout
        << "Encountered error: " << std::endl
        << error << std::endl;
    return 1;
    }

  return 0;
}
