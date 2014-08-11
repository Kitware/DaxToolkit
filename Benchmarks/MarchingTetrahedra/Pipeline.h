//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2014 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================

#include "ArgumentsParser.h"

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DispatcherGenerateInterpolatedCells.h>
#include <dax/cont/DispatcherGenerateTopology.h>
#include <dax/cont/DispatcherMapCell.h>
#include <dax/cont/DispatcherMapField.h>
#include <dax/cont/Timer.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/worklet/Magnitude.h>
#include <dax/worklet/Tetrahedralize.h>
#include <dax/worklet/MarchingTetrahedra.h>

#include <iostream>
#include <vector>
#include <fstream>

#define MAKE_STRING2(x) #x
#define MAKE_STRING1(x) MAKE_STRING2(x)
#define DEVICE_ADAPTER MAKE_STRING1(DAX_DEFAULT_DEVICE_ADAPTER_TAG)

namespace
{

dax::Scalar ISOVALUE = 100;

void PrintResults(int pipeline, double time)
{
  std::cout << "Elapsed time: " << time << " seconds." << std::endl;
  std::cout << "CSV," DEVICE_ADAPTER ","
            << pipeline << "," << time << std::endl;
}

template<typename T, typename Stream>
void PrintContentsToStream(dax::cont::UnstructuredGrid<T>& grid, Stream &stream)
  {
  std::cout << "Printing data to VTK file" << std::endl;
  const dax::Id num_points(grid.GetNumberOfPoints());
  const dax::Id num_cells(grid.GetNumberOfCells());

  //dump header
  stream << "# vtk DataFile Version 3.0"  << std::endl;
  stream << "vtk output" << std::endl;
  stream << "ASCII"  << std::endl;
  stream << "DATASET POLYDATA" << std::endl;
  stream << "POINTS " << num_points << " float" << std::endl;

  std::vector<dax::Vector3> contPoints(num_points);
  grid.GetPointCoordinates().CopyInto(contPoints.begin());

  for(dax::Id i=0; i < num_points; ++i)
    {
    dax::Vector3 coord = contPoints[i];
    stream << coord[0] << " " << coord[1] << " " << coord[2] << " ";
    if(i%3==2)
      {
      stream << std::endl; //pump new line after each 3rd vector
      }
    }
  if(num_points%3!=0)
    {
    stream << std::endl;
    }

  //print cells
  enum {NUM_VERTS = dax::CellTraits<T>::NUM_VERTICES};
  stream << "POLYGONS " << num_cells << " " << num_cells  * (NUM_VERTS+1) << std::endl;

  std::vector<dax::Id> contTopo(num_cells*NUM_VERTS);
  grid.GetCellConnections().CopyInto(contTopo.begin());

  dax::Id index=0;
  for(dax::Id i=0; i < num_cells; ++i,index+=NUM_VERTS)
    {
    stream << NUM_VERTS << " ";
    stream << contTopo[index+0] << " ";
    stream << contTopo[index+1] << " ";
    stream << contTopo[index+2] << std::endl;
    }
  stream << std::endl;
  }

template <typename InGridType, typename OutGridType>
void GridTetrahedralize(const InGridType& inGrid, OutGridType& outGrid)
  {
  const dax::Id cellConnLength = inGrid.GetNumberOfCells() * 5 * 4 ;
  std::vector<dax::Id> cellConnections(cellConnLength,-1);
  dax::cont::ArrayHandle<dax::Id> cellHandle =
          dax::cont::make_ArrayHandle(cellConnections);

    typedef dax::cont::DispatcherGenerateTopology<
                          dax::worklet::Tetrahedralize,
                          dax::cont::ArrayHandleConstant<dax::Id>
                          > DispatcherTopology;
    typedef dax::cont::ArrayHandleConstant<dax::Id> CountHandleType;

    CountHandleType count(5,inGrid.GetNumberOfCells());

    DispatcherTopology dispatcher(count);

    //Although all the points from the input go into the output, if you set
    //this to false Dax will not copy the points to the unstructured grid due
    //to a templating snafu. The easiest way to get points (albiet not the
    //fastest way) is to turn this on.
    dispatcher.SetRemoveDuplicatePoints(true);

    outGrid.SetCellConnections(cellHandle);
    dispatcher.Invoke(inGrid,outGrid);
}
void RunDAXPipeline(const dax::cont::UniformGrid<> &ugrid, int pipeline)
{
  std::cout << "Running pipeline " << pipeline << ": Magnitude -> MarchingTetrahedra" << std::endl;

  dax::cont::UnstructuredGrid<dax::CellTagTriangle> outGrid;
  dax::cont::UnstructuredGrid<dax::CellTagTetrahedron> tetGrid;

  //tetrahedralize unifrom grid
  GridTetrahedralize(ugrid, tetGrid);

  dax::cont::ArrayHandle<dax::Scalar> intermediate1;
  dax::cont::DispatcherMapField< dax::worklet::Magnitude > magDispatcher;
  magDispatcher.Invoke( tetGrid.GetPointCoordinates(), intermediate1);

  dax::cont::Timer<> timer;

  //dispatch marching tetrahedra worklet generate step
  typedef dax::cont::DispatcherGenerateInterpolatedCells< dax::worklet::MarchingTetrahedraGenerate > DispatcherIC;
  typedef DispatcherIC::CountHandleType  CountHandleType;

  dax::worklet::MarchingTetrahedraCount classifyWorklet(ISOVALUE);
  dax::worklet::MarchingTetrahedraGenerate generateWorklet(ISOVALUE);

  //run the first step
  CountHandleType count; //array handle for the first step count
  dax::cont::DispatcherMapCell<dax::worklet::MarchingTetrahedraCount > cellDispatcher( classifyWorklet );
  cellDispatcher.Invoke(tetGrid, intermediate1, count);

  //construct the topology generation worklet
  DispatcherIC icDispatcher(count, generateWorklet );
  icDispatcher.SetRemoveDuplicatePoints(
                pipeline == dax::testing::ArgumentsParser::MARCHING_TETRAHEDRA_REMOVE_DUPLICATES);

  //run the second step
  icDispatcher.Invoke(tetGrid, outGrid, intermediate1);

  double time = timer.GetElapsedTime();

  std::cout << "number of coordinates in: " << tetGrid.GetNumberOfPoints() << std::endl;
  std::cout << "number of coordinates out: " << outGrid.GetNumberOfPoints() << std::endl;
  std::cout << "number of cells out: " << outGrid.GetNumberOfCells() << std::endl;
  PrintResults(pipeline, time);

  if( time < 0)  //rough dump to file, currently disabled
    {
    std::ofstream file;
    file.open ("MTdaxResult.vtk");
    PrintContentsToStream(outGrid,file);
    file.close();
    }

}


} // Anonymous namespace

