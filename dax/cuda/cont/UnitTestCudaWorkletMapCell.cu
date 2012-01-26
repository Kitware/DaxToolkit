/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/worklet/CellGradient.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>

#include <dax/cont/internal/Testing.h>

#include <typeinfo>

namespace {

const dax::Id DIM = 64;

//-----------------------------------------------------------------------------
static void TestCellGradient()
{
  // This might be a compile error if Cuda DeviceAdapter is not selected.
  DAX_TEST_ASSERT(typeid(DAX_DEFAULT_DEVICE_ADAPTER)
                  == typeid(dax::cuda::cont::DeviceAdapterCuda),
                  "Wrong device adapter automatically selected.");

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
  dax::cont::ArrayHandle<dax::Scalar> fieldHandle(field.begin(), field.end());

  std::vector<dax::Vector3> gradient(grid.GetNumberOfCells());
  dax::cont::ArrayHandle<dax::Vector3> gradientHandle(gradient.begin(),
                                                      gradient.end());

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
                    "Got bad cosine");
    }
}

} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestCudaWorkletMapCell(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestCellGradient);
}
