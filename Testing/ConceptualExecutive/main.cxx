/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <stdio.h>
#include <iostream>

#include <vector>
#include <assert.h>

#include "RuntimeTraits.h"
#include "HeterogeneousContainer.h"


void RuntimeFields()
{
  std::cout << "RuntimeFields" << std::endl;
  Grid grid(10,20);
  Model<Grid> model(grid);

  Filter< FieldModule<int> > m(model.pointField());
  std::cout << "Filter data size is " << m.size() << std::endl;
  std::cout << "Field type is " << m.fieldType() << std::endl;
}

void ConnectFilterFields()
{
  std::cout << "ConnectFilterFields" << std::endl;
  Grid grid(10,20);
  Model<Grid> model(grid);
  {
    Filter<FieldModule <int> > Filter1(model.pointField());
    Filter<FieldModule <float> > Filter2(Filter1);

    std::cout << "Filter1 Field Type: " << Filter1.fieldType() << " and is mergeable with Parent Filter: " << Filter1.isMergeable() << std::endl;
    std::cout << "Filter2 Field Type: " << Filter2.fieldType() << " and is mergeable with Parent Filter: " << Filter2.isMergeable() << std::endl;
  }

  {
    Filter<FieldModule <double> > Filter3(model.cellField());
    Filter<FieldModule <double> > Filter4(Filter3);

    std::cout << "Filter1 Field Type: " << Filter3.fieldType() << " and is mergeable with Parent Filter: " << Filter3.isMergeable() << std::endl;
    std::cout << "Filter2 Field Type: " << Filter4.fieldType() << " and is mergeable with Parent Filter: " << Filter4.isMergeable() << std::endl;
  }
}

void ConnectPointToCell1()
{
  std::cout << "ConnectPointToCell1" << std::endl;
  Grid grid(10,20);
  Model<Grid> model(grid);

  Filter<FieldModule  <float> > Filter1(model.pointField());
  Filter<PointToCellModule  <float> > Filter2(Filter1);

  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << " and is mergeable with Parent Filter: " << Filter2.isMergeable() << std::endl;
}

void ConnectCellWithPoint()
{
  std::cout << "ConnectCellWithPoint" << std::endl;
  Grid grid(10,20);
  Model<Grid> model(grid);

  Filter<FieldModule <Grid> > Filter1(model.pointField());
  Filter<CellModuleWithPointInput <Grid> > Filter2(Filter1);

  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << " and is mergeable with Parent Filter: " << Filter2.isMergeable() << std::endl;
}

void ConnectCellWithPoint2()
{
  std::cout << "ConnectCellWithPoint2" << std::endl;
  Grid grid(10,20);
  Model<Grid> model(grid);

  Filter<FieldModule<bool> > Filter1(model.pointField());
  Filter<FieldModule<bool> > Filter2(model.cellField());
  Filter<CellModuleWithPointInput<char> > Filter3(Filter2,Filter1);

  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << " and is mergeable with Parent Filter: " << Filter2.isMergeable() << std::endl;
  std::cout << "Filter3 Field Type: " << Filter3.fieldType() << std::endl;
}

void ConnectCellToPoint()
{
  std::cout << "ConnectCellToPoint" << std::endl;
  Grid grid(10,20);
  Model<Grid> model(grid);

  Filter<FieldModule <int> > Filter1(model.cellField());
  Filter<CellToPointModule <double> > Filter2(Filter1);

  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << " and is mergeable with Parent Filter: " << Filter2.isMergeable() << std::endl;
}

void ConnectPointToCell_TO_CellToPoint()
{
  std::cout << "ConnectCellToPoint" << std::endl;
  Grid grid(10,20);
  Model<Grid> model(grid);

  Filter<FieldModule <int> > Filter1(model.pointField());
  Filter<PointToCellModule <bool> > Filter2(Filter1);
  Filter<FieldModule <int> > FilterTest(Filter2);
  Filter<CellToPointModule <float> > Filter3(FilterTest);

  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << std::endl;
  std::cout << "Filter2 Field Type: " << Filter2.fieldType() << " and is mergeable with Parent Filter: " << Filter2.isMergeable() << std::endl;
  std::cout << "Filter3 Field Type: " << Filter3.fieldType() << std::endl;
  std::cout << "FilterTest Field Type: " << FilterTest.fieldType() << std::endl;
}




void HeterogeneousContainer()
{
  std::cout << "HeterogeneousContainer" << std::endl;
  HQueue hq;
  int i = 5, j;
  double x = 9.8, y;
  hq.push(i);
  hq.push(x);

  std::cout << hq.is_type(i) << " " // implicit type queries
  << hq.is_type(x) << " "
  << hq.is_type<int>() << " " // explicit type queries
  << hq.is_type<double>() << std::endl;

  //hq.pop(y); // Error: will throw an exception
  hq.pop(j);
  hq.pop(y);
  std::cout << j << " " << y << std::endl;
}

int main(int argc, char* argv[])
{
  //HeterogeneousContainer();

  RuntimeFields();
  ConnectFilterFields();
  ConnectPointToCell1();
  ConnectCellWithPoint();
  ConnectCellWithPoint2();
  ConnectCellToPoint();
  ConnectPointToCell_TO_CellToPoint();

  return 0;
}
