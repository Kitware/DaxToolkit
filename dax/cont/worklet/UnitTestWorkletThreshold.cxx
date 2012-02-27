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

#include <dax/cont/internal/Testing.h>
#include <vector>


namespace {

const dax::Id DIM = 64;

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
  dax::Scalar min = 70;
  dax::Scalar max = 82;

  try
    {
    dax::cont::worklet::Threshold(grid,grid2,min,max,fieldHandle,resultHandle);
    }
  catch (dax::cont::ErrorControl error)
    {
    std::cout << "Got error: " << error.GetMessage() << std::endl;
    DAX_TEST_ASSERT(true==false,error.GetMessage());
    }

  std::vector<dax::Scalar> result(resultHandle.GetNumberOfEntries());
  resultHandle.SetNewControlData(result.begin(),result.end());
  resultHandle.CompleteAsOutput(); //fetch back to control

  //test max < min.
}

} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletThreshold(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestThreshold);
}

