/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <stdio.h>
#include <iostream>

#include "Filter.h"
#include "Worklets.h"
#include "StructuredGrid.h"

void createGrid(StructuredGrid *&grid)
{
  dax::Vector3 origin = dax::make_Vector3(0,0,0);
  dax::Vector3 spacing = dax::make_Vector3(.2,.2,.2);
  dax::Extent3 extents = dax::Extent3( dax::make_Id3(0,0,0), dax::make_Id3(10,10,10));
  grid = new StructuredGrid(origin,spacing,extents);
}

template <typename T, typename U>
void executePipeline(T *points, U &result)
{
  execute<worklets::Elevation>()(points,result);
  execute<worklets::Sine>()(result,result);
  execute<worklets::Square>()(result,result);
  execute<worklets::Cosine>()(result,result);
}

void TestHostPipeline()
{
  std::cout << "TestElevation Host" << std::endl;
  StructuredGrid* grid;
  createGrid(grid);

  //hand created push driven pipeline on the cpu
  dax::ScalarArray result("result");
  executePipeline(grid->points(),result);

}

void TestDevicePipeline()
{
  std::cout << "TestElevation Device" << std::endl;
  StructuredGrid* grid;
  createGrid(grid);

  //move the data to the gpu

  //hand created push driven pipeline on the gpu
  dax::DeviceScalarArray result("result");
  executePipeline(grid->points(),result);
}

void RuntimeFields()
{
  std::cout << "RuntimeFields" << std::endl;
  StructuredGrid* grid;
  createGrid(grid);

//  we create a trivial producer filter
//  Filter<worklets::Elevation> elev(grid);
//  Filter<worklets::Sine> sf(elev); //implicitly copy everything from the producer
//  Filter<worklets::Square> sqf(sf);
//  Filter<worklets::Cosine> cf(sqf);

//  instead of working on creation time, we now execute when we call run
//  cf.run();
}

int main()
{
  TestHostPipeline();
  TestDevicePipeline();

  return 0;
}
