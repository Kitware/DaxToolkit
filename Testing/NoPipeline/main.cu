/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <stdio.h>
#include <iostream>

#include <dax/cont/StructuredGrid.h>
#include <dax/cont/Array.h>

#include "Worklets.h"

dax::cont::StructuredGrid CreateInputStructure(dax::Id dim)
{
  dax::cont::StructuredGrid grid(
    dax::make_Vector3(0.0, 0.0, 0.0),
    dax::make_Vector3(1.0, 1.0, 1.0),
    dax::make_Id3(0, 0, 0),
    dax::make_Id3(dim-1, dim-1, dim-1) );

  dax::cont::Array<dax::Id>* testPData = new
    dax::cont::Array<dax::Id>();
  testPData->setName("pointArray");
  testPData->resize(grid.numPoints(),1);
  grid.addPointField(testPData);

  dax::cont::Array<dax::Id>* testCData = new
    dax::cont::Array<dax::Id>();
  testCData->setName("cellArray");
  testCData->resize(grid.numCells(),1);
  grid.addCellField(testCData);

  return grid;
}

void ConnectFilterFields()
{
  std::cout << "ConnectFilterFields" << std::endl;
  dax::cont::StructuredGrid grid = CreateInputStructure(32);
  {  
    worklets::Elevation(grid,
                        *grid.points(),
                        helpers::pointFieldHandle<dax::Scalar>("Elevation"));
    worklets::Square(grid,
                     *grid.pointField("Elevation"),
                     helpers::pointFieldHandle<dax::Scalar>("Square"));
  }

  {
    worklets::Sine(grid,
                   *grid.cellField("cellArray"),
                   helpers::cellFieldHandle<dax::Scalar>("Sine"));

    worklets::Cosine(grid,
                     *grid.cellField("Sine"),
                     helpers::cellFieldHandle<dax::Scalar>("Square"));
  }
}

void ConnectCellWithPoint()
{
  std::cout << "ConnectCellWithPoint" << std::endl;

  dax::cont::StructuredGrid grid = CreateInputStructure(32);

//  worklets::CellGradient(grid,
//                         grid.points(),
//                         grid.pointField("pointArray"),
//                         helpers::cellFieldHandle<dax::Vector3>("Gradient"));
}

int main(int argc, char* argv[])
{
  ConnectFilterFields();
  ConnectCellWithPoint();

  return 0;
}
