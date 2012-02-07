/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/DeviceAdapterDebug.h>
#include <dax/cont/internal/DeviceAdapterError.h>

#include <dax/cont/worklet/Elevation.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>

#include <dax/cont/internal/Testing.h>

namespace {

const dax::Id DIM = 64;

//-----------------------------------------------------------------------------
static void TestElevation()
{
  dax::cont::UniformGrid grid;
  grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(DIM-1, DIM-1, DIM-1));

  std::vector<dax::Scalar> elevation(grid.GetNumberOfPoints());
  dax::cont::ArrayHandle<dax::Scalar, dax::cont::DeviceAdapterDebug>
      elevationHandle(elevation.begin(), elevation.end());

  std::cout << "Running Elevation worklet" << std::endl;
  dax::cont::worklet::Elevation(grid, grid.GetPoints(), elevationHandle);

  std::cout << "Checking result" << std::endl;
  dax::Id3 ijk;
  for (ijk[2] = 0; ijk[2] < DIM; ijk[2]++)
    {
    for (ijk[1] = 0; ijk[1] < DIM; ijk[1]++)
      {
      for (ijk[0] = 0; ijk[0] < DIM; ijk[0]++)
        {
        dax::Id pointIndex = grid.ComputePointIndex(ijk);
        dax::Scalar elevationValue = elevation[pointIndex];
        dax::Vector3 pointCoordinates = grid.GetPointCoordinates(pointIndex);
        // Wrong, but what is currently computed.
        dax::Scalar elevationExpected
            = sqrt(dax::dot(pointCoordinates, pointCoordinates));
        DAX_TEST_ASSERT(test_equal(elevationValue, elevationExpected),
                        "Got bad elevation.");
        }
      }
    }
}

} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletElevation(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestElevation);
}
