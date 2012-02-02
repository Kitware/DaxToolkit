/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/worklet/testing/FieldMapError.h>

#include <dax/TypeTraits.h>
#include <dax/cont/ErrorExecution.h>
#include <dax/cont/UniformGrid.h>

#include <dax/cont/internal/Testing.h>

namespace {

const dax::Id DIM = 64;

//-----------------------------------------------------------------------------
static void TestFieldMapError()
{
  dax::cont::UniformGrid grid;
  grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(DIM-1, DIM-1, DIM-1));

  std::cout << "Running field map worklet that errors" << std::endl;
  bool gotError = false;
  try
    {
    dax::cont::worklet::testing::FieldMapError
        <dax::cont::UniformGrid, dax::cont::DeviceAdapterDebug>(grid);
    }
  catch (dax::cont::ErrorExecution error)
    {
    std::cout << "Got expected ErrorExecution object." << std::endl;
    std::cout << error.GetMessage() << std::endl;
    gotError = true;
    }

  DAX_TEST_ASSERT(gotError, "Never got the error thrown.");
}

} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletMapFieldError(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestFieldMapError);
}
