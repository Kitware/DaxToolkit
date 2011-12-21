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
#include <dax/cont/Worklets.h>
#include <dax/cont/FieldHandles.h>


void CreateInputStructure(dax::Id dim,dax::cont::StructuredGrid &grid )
  {

  grid.Origin = dax::make_Vector3(0.0, 0.0, 0.0);
  grid.Spacing = dax::make_Vector3(1.0, 1.0, 1.0),
      grid.Extent.Min = dax::make_Id3(0, 0, 0),
      grid.Extent.Max = dax::make_Id3(dim-1, dim-1, dim-1);

  dax::cont::Array<dax::Scalar>* testPData = new
                                             dax::cont::Array<dax::Scalar>();

  testPData->resize(grid.numPoints(),1);
  grid.getFieldsPoint().addArray("pointArray",testPData);

  dax::cont::Array<dax::Scalar>* testCData = new
                                             dax::cont::Array<dax::Scalar>();
  testCData->resize(grid.numCells(),1);

  grid.getFieldsCell().addArray("cellArray",testCData);

  grid.computePointLocations();
  }

void ConnectFilterFields()
  {
  std::cout << "ConnectFilterFields" << std::endl;
  dax::cont::StructuredGrid grid;
  CreateInputStructure(128,grid);
  {
  std::cout << "Elevation" << std::endl;
  dax::cont::worklets::Elevation(grid,
                                 grid.points(),
                                 dax::cont::pointFieldHandle<dax::Scalar>("Elevation"));
  std::cout << "Square" << std::endl;
  dax::cont::worklets::Square(grid,
                              grid.getFieldsPoint().getScalar("Elevation"),
                              dax::cont::pointFieldHandle<dax::Scalar>("Square"));
  }

  {
  std::cout << "Square" << std::endl;
  dax::cont::worklets::Square(grid,
                              grid.getFieldsCell().getScalar("cellArray"),
                              dax::cont::cellFieldHandle<dax::Scalar>("Sine"));

  std::cout << "Cosine" << std::endl;
  dax::cont::worklets::Cosine(grid,
                              grid.getFieldsCell().getScalar("Sine"),
                              dax::cont::cellFieldHandle<dax::Scalar>("Square"));
  }
  }

void ConnectCellWithPoint()
  {
  std::cout << "ConnectCellWithPoint" << std::endl;

  dax::cont::StructuredGrid grid;
  CreateInputStructure(128,grid);

  std::cout << "Gradient" << std::endl;
  dax::cont::worklets::CellGradient(grid,
                                    grid.points(),
                                    grid.getFieldsPoint().getScalar("pointArray"),
                                    dax::cont::cellFieldHandle<dax::Vector3>("Gradient"));

  std::cout << "Elevation" << std::endl;
  dax::cont::worklets::Elevation(grid,
                                 grid.getFieldsCell().getVector3("Gradient"),
                                 dax::cont::cellFieldHandle<dax::Scalar>("Elev"));
  }

void Pipeline1Test()
  {
  std::cout << "Pipeline1" << std::endl;
  dax::cont::StructuredGrid grid;
  CreateInputStructure(128,grid);

  std::cout << "Elevation" << std::endl;
  dax::cont::worklets::Elevation(grid,
                                 grid.points(),
                                 dax::cont::pointFieldHandle<dax::Scalar>("Elevation"));


  std::cout << "Gradient" << std::endl;
  dax::cont::worklets::CellGradient(grid,
                                    grid.points(),
                                    grid.getFieldsPoint().getScalar("Elevation"),
                                    dax::cont::cellFieldHandle<dax::Vector3>("Gradient"));
  }

int main(int argc, char* argv[])
  {
  ConnectFilterFields();
  ConnectCellWithPoint();
  Pipeline1Test();

  return 0;
  }
