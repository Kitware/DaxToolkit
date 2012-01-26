/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/worklet/Square.h>

#include <dax/TypeTraits.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>

#include <dax/cont/internal/Testing.h>

#include <typeinfo>

namespace {

const dax::Id DIM = 64;

//-----------------------------------------------------------------------------
static void TestSquare()
{
  // This might be a compile error if OpenMP DeviceAdapter is not selected.
  DAX_TEST_ASSERT(typeid(DAX_DEFAULT_DEVICE_ADAPTER)
                  == typeid(dax::openmp::cont::DeviceAdapterOpenMP),
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

  std::vector<dax::Scalar> square(grid.GetNumberOfPoints());
  dax::cont::ArrayHandle<dax::Scalar> squareHandle(square.begin(),
                                                   square.end());

  std::cout << "Running Square worklet" << std::endl;
  dax::cont::worklet::Square(grid, fieldHandle, squareHandle);

  std::cout << "Checking result" << std::endl;
  for (dax::Id pointIndex = 0;
       pointIndex < grid.GetNumberOfPoints();
       pointIndex++)
    {
    dax::Scalar squareValue = square[pointIndex];
    dax::Scalar squareTrue = field[pointIndex]*field[pointIndex];
    DAX_TEST_ASSERT(test_equal(squareValue, squareTrue),
                    "Got bad square");
    }
}

} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestOpenMPWorkletMapField(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestSquare);
}
