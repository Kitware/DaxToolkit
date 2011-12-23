/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <stdio.h>
#include <iostream>

#include <dax/cont/StructuredGrid.h>
#include <dax/cont/Array.h>

//has to be above dax/cont/Worklets
#include <dax/cuda/cont/Worklets.h>

//should we have a commoon all header for control?
#include <dax/cont/Worklets.h>
#include <dax/cont/FieldHandles.h>

void CreateInputStructure(dax::Id dim,dax::cont::StructuredGrid &grid )
  {

  grid.Origin = dax::make_Vector3(0.0, 0.0, 0.0);
  grid.Spacing = dax::make_Vector3(1.0, 1.0, 1.0),
      grid.Extent.Min = dax::make_Id3(0, 0, 0),
      grid.Extent.Max = dax::make_Id3(dim-1, dim-1, dim-1);

  dax::cont::ArrayPtr<dax::Scalar> testPData(
        new dax::cont::Array<dax::Scalar>());

  testPData->resize(grid.numPoints(),1);
  grid.getFieldsPoint().addArray("pointArray",testPData);

  dax::cont::ArrayPtr<dax::Scalar> testCData(
        new dax::cont::Array<dax::Scalar>());
  testCData->resize(grid.numCells(),1);

  grid.getFieldsCell().addArray("cellArray",testCData);

  grid.computePointLocations();
  }

void PrintCheckValues(const dax::cont::ArrayPtr<dax::Vector3> &array)
{
  for (dax::Id index = 0; index < array->size(); index++)
    {
    dax::Vector3 value = (*array)[index];
    if (index < 20)
      {
      std::cout << index << " : " << value.x << ", " << value.y << ", " << value.z
           << std::endl;
      }
    if (   (value.x < -1) || (value .x > 1)
        || (value.y < -1) || (value .y > 1)
        || (value.z < -1) || (value .z > 1) )
      {
      std::cout << index << " : " << value.x << ", " << value.y << ", " << value.z
           << std::endl;
      break;
      }
    }
}

void ConnectFilterFields()
  {
  std::cout << "ConnectFilterFields" << std::endl;
  dax::cont::StructuredGrid grid;
  CreateInputStructure(32,grid);
  {  
  dax::cont::worklets::Elevation(grid,
                                 grid.points(),
                                 dax::cont::pointFieldHandle<dax::Scalar>("Elevation"));
  dax::cont::worklets::Elevation(grid,
                                 grid.points(),
                                 dax::cont::pointFieldHandle<dax::Scalar>("Elevation2"));
  dax::cont::worklets::Elevation(grid,
                                 grid.points(),
                                 dax::cont::pointFieldHandle<dax::Scalar>("Elevation3"));

  dax::cont::worklets::Square(grid,
                              grid.getFieldsPoint().getScalar("Elevation"),
                              dax::cont::pointFieldHandle<dax::Scalar>("Square"));
  }

  {
  dax::cont::worklets::Square(grid,
                              grid.getFieldsCell().getScalar("cellArray"),
                              dax::cont::cellFieldHandle<dax::Scalar>("Square"));


  dax::cont::worklets::Cosine(grid,
                              grid.getFieldsCell().getScalar("Square"),
                              dax::cont::cellFieldHandle<dax::Scalar>("Sine"));
  }
  }

void ConnectCellWithPoint()
  {
  std::cout << "ConnectCellWithPoint" << std::endl;

  dax::cont::StructuredGrid grid;
  CreateInputStructure(32,grid);

  dax::cont::worklets::CellGradient(grid,
                                    grid.points(),
                                    grid.getFieldsPoint().getScalar("pointArray"),
                                    dax::cont::cellFieldHandle<dax::Vector3>("Gradient"));

  dax::cont::worklets::Elevation(grid,
                                 grid.getFieldsCell().getVector3("Gradient"),
                                 dax::cont::cellFieldHandle<dax::Scalar>("Elev"));
  }

void Pipeline1Test()
  {
  std::cout << "Pipeline1" << std::endl;
  dax::cont::StructuredGrid grid;
  CreateInputStructure(32,grid);

  dax::cont::worklets::Elevation(grid,
                                 grid.points(),
                                 dax::cont::pointFieldHandle<dax::Scalar>("Elevation"));

  dax::cont::worklets::CellGradient(grid,
                                    grid.points(),
                                    grid.getFieldsPoint().getScalar("Elevation"),
                                    dax::cont::cellFieldHandle<dax::Vector3>("Gradient"));

  //you need use the retrieve function
  //to get the propery Array, don't attempt to get the array
  //directly ( TODO remove the ability to get raw array from public control API)
  dax::cont::ArrayPtr<dax::Vector3> array = dax::cont::retrieve(
                                  grid.getFieldsCell().getVector3("Gradient"));

  PrintCheckValues(array);
  }

int main(int argc, char* argv[])
  {
  //ConnectFilterFields();
  //ConnectCellWithPoint();
  Pipeline1Test();

  return 0;
  }
