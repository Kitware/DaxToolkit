/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/worklet/Threshold.h>

#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <dax/TypeTraits.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

#include <vector>

namespace {

const dax::Id DIM = 64;
const dax::Scalar TOLERANCE = 0.0001;

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
static void TestThreshold()
{
  dax::cont::UniformGrid grid;
  grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(DIM-1, DIM-1, DIM-1));

  dax::cont::UnstructuredGrid<dax::exec::CellHexahedron> grid2;
  dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);

  std::vector<dax::Scalar> field(grid.GetNumberOfPoints());
  for (dax::Id pointIndex = 0;
       pointIndex < grid.GetNumberOfPoints();
       pointIndex++)
    {
    field[pointIndex]
        = dax::dot(grid.GetPointCoordinates(pointIndex), trueGradient);
    }
  dax::cont::ArrayHandle<dax::Scalar> fieldHandle(field.begin(), field.end());

  //unkown size
  dax::cont::ArrayHandle<dax::Scalar> resultHandle;

  std::cout << "Running Threshold worklet" << std::endl;
  dax::Scalar min = 0.1;
  dax::Scalar max = 0.2;
  dax::cont::worklet::Threshold(grid,grid2,min,max,fieldHandle,resultHandle);

  std::vector<dax::Scalar> result(resultHandle.GetNumberOfEntries());
  resultHandle.SetNewControlData(result.begin(),result.end());
  resultHandle.CompleteAsOutput(); //fetch bac to control
}

} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletThreshold(int, char *[])
{
  try
    {
    TestThreshold();
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
