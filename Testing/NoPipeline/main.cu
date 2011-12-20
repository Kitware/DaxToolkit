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

void CreateInputStructure(dax::Id dim,dax::cont::StructuredGrid &grid )
{

  grid.Origin = dax::make_Vector3(0.0, 0.0, 0.0);
  grid.Spacing = dax::make_Vector3(1.0, 1.0, 1.0),
  grid.Extent.Min = dax::make_Id3(0, 0, 0),
  grid.Extent.Max = dax::make_Id3(dim-1, dim-1, dim-1);

  dax::cont::Array<dax::Scalar>* testPData = new
    dax::cont::Array<dax::Scalar>();

  testPData->resize(grid.numPoints(),1);
  grid.getFieldsPoint().addArray("pointArray",*testPData);

  dax::cont::Array<dax::Scalar>* testCData = new
    dax::cont::Array<dax::Scalar>();
  testCData->resize(grid.numCells(),1);

  grid.getFieldsCell().addArray("cellArray",*testCData);

  grid.computePointLocations();
}

void ConnectFilterFields()
{
  std::cout << "ConnectFilterFields" << std::endl;
  dax::cont::StructuredGrid grid;
  CreateInputStructure(32,grid);
  {  
    std::cout << "Elevation" << std::endl;
    worklets::Elevation(grid,
                        grid.points(),
                        helpers::pointFieldHandle<dax::Scalar>("Elevation"));
    std::cout << "Square" << std::endl;
    worklets::Square(grid,
                     grid.getFieldsPoint().getScalar("Elevation"),
                     helpers::pointFieldHandle<dax::Scalar>("Square"));
  }

  {
    std::cout << "Square" << std::endl;
    worklets::Square(grid,
                   grid.getFieldsCell().getScalar("cellArray"),
                   helpers::cellFieldHandle<dax::Scalar>("Sine"));

    std::cout << "Cosine" << std::endl;
    worklets::Cosine(grid,
                     grid.getFieldsCell().getScalar("Sine"),
                     helpers::cellFieldHandle<dax::Scalar>("Square"));
  }
}

void ConnectCellWithPoint()
{
  std::cout << "ConnectCellWithPoint" << std::endl;

  dax::cont::StructuredGrid grid;
  CreateInputStructure(32,grid);

  worklets::CellGradient(grid,
                         grid.points(),
                         grid.getFieldsPoint().getScalar("pointArray"),
                         helpers::cellFieldHandle<dax::Vector3>("Gradient"));

  std::cout << "Elevation" << std::endl;
  worklets::Elevation(grid,
                      grid.getFieldsCell().getVector3("Gradient"),
                      helpers::cellFieldHandle<dax::Scalar>("Elev"));
}

int main(int argc, char* argv[])
{
  try
   {
    ConnectFilterFields();
   }
   catch(thrust::system_error e)
   {
     std::cerr  << "Error: " << e.what() << std::endl;
     exit(-1);
   }

  //ConnectCellWithPoint();

  return 0;
}
