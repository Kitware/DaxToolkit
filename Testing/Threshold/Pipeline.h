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
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================

#include <stdio.h>
#include <iostream>
#include "Timer.h"

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/worklet/Elevation.h>
#include <dax/cont/worklet/Threshold.h>

#include <vector>
#include <iostream>
#include <fstream>

#define MAKE_STRING2(x) #x
#define MAKE_STRING1(x) MAKE_STRING2(x)
#define DEVICE_ADAPTER MAKE_STRING1(DAX_DEFAULT_DEVICE_ADAPTER)

namespace
{

class CheckValid {
public:
  CheckValid() : Valid(true) { }
  operator bool() { return this->Valid; }
  void operator()(dax::Scalar value) {
    if ((value < 0) || (value > 100)) { this->Valid = false; }
  }
private:
  bool Valid;
};

void PrintScalarValue(dax::Scalar value)
{
  std::cout << " " << value;
}


template<class IteratorType>
void CheckValues(IteratorType begin, IteratorType end)
{
  typedef typename std::iterator_traits<IteratorType>::value_type VectorType;

  CheckValid isValid;
  for (IteratorType iter = begin; iter != end; iter++)
    {
    VectorType vector = *iter;
    dax::cont::VectorForEach(vector, isValid);
    if (!isValid)
      {
      std::cout << "*** Encountered bad value." << std::endl;
      std::cout << std::distance(begin,iter) << ":";
      dax::cont::VectorForEach(vector, PrintScalarValue);
      std::cout << std::endl;
      break;
      }
    }
}

void PrintResults(int pipeline, double time)
{
  std::cout << "Elapsed time: " << time << " seconds." << std::endl;
  std::cout << "CSV," DEVICE_ADAPTER ","
            << pipeline << "," << time << std::endl;
}

template<typename T, typename Stream>
void PrintContentsToStream(dax::cont::UnstructuredGrid<T>& grid, Stream &stream)
  {
  const dax::Id num_points(grid.GetNumberOfPoints());
  const dax::Id num_cells(grid.GetNumberOfCells());

  //dump header
  stream << "# vtk DataFile Version 3.0"  << std::endl;
  stream << "vtk output" << std::endl;
  stream << "ASCII"  << std::endl;
  stream << "DATASET UNSTRUCTURED_GRID" << std::endl;
  stream << "POINTS " << num_points << " float" << std::endl;

  std::vector<dax::Vector3> contPoints(num_points);
  grid.GetCoordinatesHandle().SetNewControlData(contPoints.begin(),contPoints.end());
  grid.GetCoordinatesHandle().CompleteAsOutput();

  for(dax::Id i=0; i < num_points; ++i)
    {
    dax::Vector3 coord = contPoints[i];
    stream << coord[0] << " " << coord[1] << " " << coord[2] << " ";
    if(i%3==2)
      {
      stream << std::endl; //pump new line after each 3rd vector
      }
    }
  if(num_points%3==2)
    {
    stream << std::endl;
    }

  //print cells
  stream << "CELLS " << num_cells << " " << num_cells  * (T::NUM_POINTS+1) << std::endl;

  std::vector<dax::Id> contTopo(num_cells*T::NUM_POINTS);
  grid.GetTopologyHandle().SetNewControlData(contTopo.begin(),contTopo.end());
  grid.GetTopologyHandle().CompleteAsOutput();

  dax::Id index=0;
  for(dax::Id i=0; i < num_cells; ++i,index+=8)
    {
    stream << T::NUM_POINTS << " ";
    stream << contTopo[index+0] << " ";
    stream << contTopo[index+1] << " ";
    stream << contTopo[index+3] << " ";
    stream << contTopo[index+2] << " ";
    stream << contTopo[index+4] << " ";
    stream << contTopo[index+5] << " ";
    stream << contTopo[index+7] << " ";
    stream << contTopo[index+6] << " ";

    stream << std::endl;
    }
  stream << std::endl;
  stream << "CELL_TYPES " << num_cells << std::endl;
  for(dax::Id i=0; i < num_cells; ++i)
    {
    stream << "11" << std::endl; //11 is voxel && 12 is hexa
    }
  }

void RunDAXPipeline(const dax::cont::UniformGrid &grid)
{
  std::cout << "Running pipeline 1: Elevation -> Threshold" << std::endl;

  dax::cont::UnstructuredGrid<dax::exec::CellHexahedron> grid2;

  dax::cont::ArrayHandle<dax::Scalar> intermediate1(grid.GetNumberOfPoints());
  dax::cont::ArrayHandle<dax::Scalar> resultHandle;

  dax::Scalar min = 0;
  dax::Scalar max = 100;

  dax::cont::worklet::Elevation(grid, grid.GetPoints(), intermediate1);

  Timer timer;
  dax::cont::worklet::Threshold(grid,grid2,min,max,intermediate1,resultHandle);
  double time = timer.elapsed();
  std::cout << "original GetNumberOfCells: " << grid.GetNumberOfCells() << std::endl;
  std::cout << "threshold GetNumberOfCells: " << grid2.GetNumberOfCells() << std::endl;

  std::cout << "original GetNumberOfPoints: " << grid.GetNumberOfPoints() << std::endl;
  std::cout << "threshold GetNumberOfPoints: " << grid2.GetNumberOfPoints() << std::endl;
  PrintResults(1, time);

  std::vector<dax::Scalar> resultsBuffer(resultHandle.GetNumberOfEntries());
  resultHandle.SetNewControlData(resultsBuffer.begin(),resultsBuffer.end());
  resultHandle.CompleteAsOutput(); //fetch back to control



  //rough dump to file
  std::ofstream file;
  file.open ("daxResult.vtk");
  PrintContentsToStream(grid2,file);
  file.close();

  CheckValues(resultsBuffer.begin(), resultsBuffer.end());
}


} // Anonymous namespace

