/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <stdio.h>
#include <iostream>

#include <vector>
#include <assert.h>

#include <dax/cont/StructuredGrid.h>
#include <dax/cont/HostArray.h>
#include "Concept.h"
#include "Worklets.h"


typedef FieldModule<worklets::Cosine> Cosine;
typedef FieldModule<worklets::Sine> Sine;
typedef FieldModule<worklets::Square> Square;
typedef FieldModule<worklets::Elevation> Elevation;

typedef CellModuleWithPointInput<worklets::CellGradient> CellGradient;

typedef PointToCellModule<worklets::PointToCell> PointToCell;
typedef CellToPointModule<worklets::CellToPoint> CellToPoint;

typedef ChangeDataModule<worklets::ChangeTopology> NewGridOutput;

dax::cont::StructuredGrid CreateInputStructure(dax::Id dim)
{
  dax::cont::StructuredGrid grid(
    dax::make_Vector3(0.0, 0.0, 0.0),
    dax::make_Vector3(1.0, 1.0, 1.0),
    dax::make_Id3(0, 0, 0),
    dax::make_Id3(dim-1, dim-1, dim-1) );

  dax::cont::HostArray<dax::Id>* testPData = new
    dax::cont::HostArray<dax::Id>();
  testPData->setName("pointArray");
  testPData->resize(grid.numPoints(),1);
  grid.addPointField(testPData);

  dax::cont::HostArray<dax::Id>* testCData = new
    dax::cont::HostArray<dax::Id>();
  testCData->setName("cellArray");
  testCData->resize(grid.numCells(),1);
  grid.addCellField(testCData);

  return grid;
}

void RuntimeFields()
{
  std::cout << "RuntimeFields" << std::endl;
  dax::cont::StructuredGrid grid = CreateInputStructure(32);
  Model<dax::cont::StructuredGrid> model(grid);

  Filter< Elevation > m(model.points());
  std::cout << "Field type is " << m.fieldType() << std::endl;
  m.run();
}

void ConnectFilterFields()
{
  std::cout << "ConnectFilterFields" << std::endl;
  dax::cont::StructuredGrid grid = CreateInputStructure(32);
  Model<dax::cont::StructuredGrid> model(grid);
  {
    Filter< Elevation > Filter1(model.points());
    Filter< Square > Filter2(Filter1);
    Filter2.run();

    std::cout << "Filter1 Field Type: " << Filter1.fieldType() << std::endl;
    std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
  }

  {
    Filter< Sine > Filter3(model.cellField("cellArray"));
    Filter< Cosine > Filter4(Filter3);
    Filter4.run();

    std::cout << "Filter3 Field Type: " << Filter3.fieldType() << std::endl;
    std::cout << "Filter4 Field Type: " << Filter4.fieldType() << std::endl;
  }
}

void ConnectPointToCell1()
{
  std::cout << "ConnectPointToCell1" << std::endl;
  dax::cont::StructuredGrid grid = CreateInputStructure(32);
  Model<dax::cont::StructuredGrid> model(grid);

  Filter< Elevation > Filter1(model.points());
  Filter< PointToCell > Filter2(Filter1);

  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
}

void ConnectCellWithPoint()
{
  std::cout << "ConnectCellWithPoint" << std::endl;
  dax::cont::StructuredGrid grid = CreateInputStructure(32);
  Model<dax::cont::StructuredGrid> model(grid);

  Filter< Elevation > Filter1(model.points());
  Filter< CellGradient > Filter2(model.cellField("cellArray"),Filter1.outputPort(0));

  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
}

void ConnectCellWithPoint2()
{
  std::cout << "ConnectCellWithPoint2" << std::endl;
  dax::cont::StructuredGrid grid = CreateInputStructure(32);
  Model<dax::cont::StructuredGrid> model(grid);

  Filter< Cosine > Filter1(model.pointField("pointArray"));
  Filter< Square > Filter2(model.cellField("cellArray"));
  Filter< CellGradient > Filter3(Filter2,Filter1);

  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
  std::cout << "Filter3 Field Type: " << Filter3.fieldType() << std::endl;
}

void ConnectCellToPoint()
{
  std::cout << "ConnectCellToPoint" << std::endl;
  dax::cont::StructuredGrid grid = CreateInputStructure(32);
  Model<dax::cont::StructuredGrid> model(grid);

  Filter< Cosine > Filter1(model.cellField("cellArray"));
  Filter< CellToPoint > Filter2(Filter1);

  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
}

void ConnectPointToCell_TO_CellToPoint()
{
  std::cout << "ConnectCellToPoint" << std::endl;
  dax::cont::StructuredGrid grid = CreateInputStructure(32);
  Model<dax::cont::StructuredGrid> model(grid);

  Filter< Square > Filter1(model.pointField("pointArray"));
  Filter< PointToCell > Filter2(Filter1);
  Filter< Sine > FilterTest(Filter2);
  Filter< CellToPoint > Filter3(FilterTest);

  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
  std::cout << "Filter3 Field Type: " << Filter3.fieldType() << std::endl;
  std::cout << "FilterTest Field Type: " << FilterTest.fieldType() << std::endl;
}

void ChangeGrid()
{
  std::cout << "ConnectCellToPoint" << std::endl;
  dax::cont::StructuredGrid grid = CreateInputStructure(32);
  Model<dax::cont::StructuredGrid> model(grid);

  Filter< Square > Filter1(model.points());
  Filter< NewGridOutput > Filter2(Filter1);
  Filter< Sine > FilterTest(Filter2);

  std::cout << "Filter1 Field Type: " << Filter1.fieldType() << std::endl;
  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
  std::cout << "FilterTest Field Type: " << FilterTest.fieldType() << std::endl;

  std::cout << "Filter1 array length " << Filter1.size() << std::endl;
  std::cout << "FilterTest array length " << FilterTest.size() << std::endl;

  Filter1.run();
  Filter2.run();
  FilterTest.run();

  std::cout << "Filter1 array length " << Filter1.size() << std::endl;
  std::cout << "FilterTest array length " << FilterTest.size() << std::endl;
}


int main(int argc, char* argv[])
{
//  RuntimeFields();
//  ConnectFilterFields();
//  ConnectPointToCell1();
//  ConnectCellWithPoint();
//  ConnectCellWithPoint2();
//  ConnectCellToPoint();
//  ConnectPointToCell_TO_CellToPoint();

  ChangeGrid();


  return 0;
}
