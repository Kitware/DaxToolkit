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

#include <dax/CellTag.h>
#include <dax/CellTraits.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DispatcherMapField.h>
#include <dax/cont/DispatcherMapCell.h>
#include <dax/cont/DispatcherGenerateTopology.h>
#include <dax/cont/Timer.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/worklet/Magnitude.h>
#include <dax/worklet/Threshold.h>

#include <vector>
#include <iostream>
#include <fstream>

#define MAKE_STRING2(x) #x
#define MAKE_STRING1(x) MAKE_STRING2(x)
#define DEVICE_ADAPTER MAKE_STRING1(DAX_DEFAULT_DEVICE_ADAPTER_TAG)

namespace
{

dax::Scalar THRESHOLD_MIN = 0;
dax::Scalar THRESHOLD_MAX = 100;

class CheckValid {
public:
  CheckValid() : Valid(true) { }
  operator bool() { return this->Valid; }
  void operator()(dax::Scalar value) {
    if ((value < THRESHOLD_MIN) || (value > THRESHOLD_MAX))
      {
      this->Valid = false;
      }
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
      exit(1);
      }
    }
}

template<typename T, class Container, class Device>
void CheckValues(const dax::cont::ArrayHandle<T,Container,Device> &array)
{
  CheckValues(array.GetPortalConstControl().GetIteratorBegin(),
              array.GetPortalConstControl().GetIteratorEnd());
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
  stream << "CELLS " << num_cells << " " << num_cells  * (NUM_VERTS+1) << std::endl;

  std::vector<dax::Id> contTopo(num_cells*NUM_VERTS);
  grid.GetCellConnections().CopyInto(contTopo.begin());

  dax::Id index=0;
  for(dax::Id i=0; i < num_cells; ++i,index+=NUM_VERTS)
    {
    stream << NUM_VERTS << " ";
    stream << contTopo[index+0] << " ";
    stream << contTopo[index+1] << " ";
    stream << contTopo[index+2] << " ";
    stream << contTopo[index+3] << " ";
    stream << contTopo[index+4] << " ";
    stream << contTopo[index+5] << " ";
    stream << contTopo[index+6] << " ";
    stream << contTopo[index+7] << std::endl;
    }
  stream << std::endl;
  stream << "CELL_TYPES " << num_cells << std::endl;
  for(dax::Id i=0; i < num_cells; ++i)
    {
    stream << "12" << std::endl; //11 is voxel && 12 is hexa
    }
  }

void RunDAXPipeline(const dax::cont::UniformGrid<> &grid)
{
  std::cout << "Running pipeline 1: Magnitude -> Threshold" << std::endl;

  dax::cont::UnstructuredGrid<dax::CellTagHexahedron> grid2;

  dax::cont::ArrayHandle<dax::Scalar> intermediate1;
  dax::cont::ArrayHandle<dax::Scalar> resultHandle;

  dax::cont::DispatcherMapField<dax::worklet::Magnitude>().Invoke(
        grid.GetPointCoordinates(),
        intermediate1);

  dax::cont::Timer<> timer;

  typedef dax::worklet::ThresholdTopology ThresholdTopologyType;
  typedef dax::worklet::ThresholdCount<dax::Scalar> ThresholdCountType;

  dax::cont::ArrayHandle<dax::Id> count;
  dax::cont::DispatcherMapCell< ThresholdCountType > clasifyDispatcher
        ( ThresholdCountType(THRESHOLD_MIN,THRESHOLD_MAX) );

  clasifyDispatcher.Invoke(grid, intermediate1, count);

  dax::cont::DispatcherGenerateTopology< ThresholdTopologyType >
        topoDispatcher(count);
  //topoDispatcher.SetRemoveDuplicatePoints(false);
  topoDispatcher.Invoke(grid,grid2);
  topoDispatcher.CompactPointField(intermediate1,resultHandle);


  double time = timer.GetElapsedTime();
  std::cout << "original GetNumberOfCells: " << grid.GetNumberOfCells() << std::endl;
  std::cout << "threshold GetNumberOfCells: " << grid2.GetNumberOfCells() << std::endl;

  std::cout << "original GetNumberOfPoints: " << grid.GetNumberOfPoints() << std::endl;
  std::cout << "threshold GetNumberOfPoints: " << grid2.GetNumberOfPoints() << std::endl;
  PrintResults(1, time);

  if(time < 0) //rough dump to file, currently disabled
    {
    std::ofstream file;
    file.open ("ThresholdDaxResult.vtk");
    PrintContentsToStream(grid2,file);
    file.close();
    }

  CheckValues(resultHandle);
}


} // Anonymous namespace

