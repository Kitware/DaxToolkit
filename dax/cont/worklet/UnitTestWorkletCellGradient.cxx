/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/DeviceAdapterDebug.h>
#include <dax/cont/internal/DeviceAdapterError.h>

#include <dax/cont/worklet/CellGradient.h>
#include <dax/TypeTraits.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>

#include <dax/cont/internal/Testing.h>

#include <math.h>

namespace {

const dax::Id DIM = 64;

//-----------------------------------------------------------------------------
static void TestCellGradient()
{
  dax::cont::UniformGrid grid;
  grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(DIM-1, DIM-1, DIM-1));

  dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);

  std::vector<dax::Scalar> field(grid.GetNumberOfPoints());
  for (dax::Id pointIndex = 0;
       pointIndex < grid.GetNumberOfPoints();
       pointIndex++)
    {
    field[pointIndex]
        = dax::dot(grid.GetPointCoordinates(pointIndex), trueGradient);
    }
  dax::cont::ArrayHandle<dax::Scalar, dax::cont::DeviceAdapterDebug>
      fieldHandle(field.begin(), field.end());

  std::vector<dax::Vector3> gradient(grid.GetNumberOfCells());
  dax::cont::ArrayHandle<dax::Vector3, dax::cont::DeviceAdapterDebug>
      gradientHandle(gradient.begin(), gradient.end());

  std::cout << "Running CellGradient worklet" << std::endl;
  dax::cont::worklet::CellGradient(grid,
                                   grid.GetPoints(),
                                   fieldHandle,
                                   gradientHandle);

  std::cout << "Checking result" << std::endl;
  for (dax::Id cellIndex = 0;
       cellIndex < grid.GetNumberOfCells();
       cellIndex++)
    {
    dax::Vector3 gradientValue = gradient[cellIndex];
    DAX_TEST_ASSERT(test_equal(gradientValue, trueGradient),
                    "Got bad gradient");
    }
}

} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletCellGradient(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestCellGradient);
}
