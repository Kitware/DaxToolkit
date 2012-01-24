/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/worklet/testing/Assert.h>

#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <dax/TypeTraits.h>

#include <dax/cont/ErrorExecution.h>
#include <dax/cont/UniformGrid.h>

#include <vector>

namespace {

const dax::Id DIM = 64;

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
static void TestAssert()
{
  dax::cont::UniformGrid grid;
  grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(DIM-1, DIM-1, DIM-1));

  std::cout << "Running field map worklet that errors" << std::endl;
  bool gotError = false;
  try
    {
    dax::cont::worklet::testing::Assert
        <dax::cont::UniformGrid, dax::cont::DeviceAdapterDebug>(grid);
    }
  catch (dax::cont::ErrorExecution error)
    {
    std::cout << "Got expected ErrorExecution object." << std::endl;
    std::cout << error.GetMessage() << std::endl;
    test_assert(error.GetWorkletName() == "Assert",
                "Got wrong worklet name.");
    gotError = true;
    }

  test_assert(gotError, "Never got the error thrown.");
}

} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletAssert(int, char *[])
{
  try
    {
    TestAssert();
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
