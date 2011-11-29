/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include "ArgumentsParser.h"

#include <cuda.h>

// Includes for host code.
#include <dax/cuda/cont/internal/DeviceArray.h>
#include <dax/cont/HostArray.h>

// Includes for device code.
#include <dax/internal/DataArray.h>
#include <dax/internal/GridStructures.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapCell.h>
#include <dax/exec/WorkMapField.h>
#include <dax/exec/internal/FieldBuild.h>
#include <dax/cuda/exec/ExecutionEnvironment.h>

//#include <Worklets/CellAverage.worklet>
#include <Worklets/CellGradient.worklet>
#include <Worklets/Cosine.worklet>
#include <Worklets/Elevation.worklet>
//#include <Worklets/PointDataToCellData.worklet>
#include <Worklets/Sine.worklet>
#include <Worklets/Square.worklet>

#include <boost/progress.hpp>


__global__ void ExecuteElevation(
    dax::internal::StructureUniformGrid grid,
    dax::internal::DataArray<dax::Scalar> outPointScalars)
{
  dax::Id pointIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
  dax::Id pointIncrement = gridDim.x;
  dax::Id numPoints = dax::internal::numberOfPoints(grid);

  dax::exec::WorkMapField<dax::exec::CellVoxel> work(grid, pointIndex);
  dax::exec::FieldCoordinates pointCoord
      = dax::exec::internal::fieldCoordinatesBuild(grid);
  dax::exec::FieldPoint<dax::Scalar> outField(outPointScalars);

  for ( ; pointIndex < numPoints; pointIndex += pointIncrement)
    {
    work.SetIndex(pointIndex);
    Elevation(work, pointCoord, outField);
    }
}

__global__ void ExecutePipeline1(
    dax::internal::StructureUniformGrid grid,
    dax::internal::DataArray<dax::Scalar> inPointScalars,
    dax::internal::DataArray<dax::Vector3> outCellVectors)
{
  dax::Id cellIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
  dax::Id cellIncrement = gridDim.x;
  dax::Id numCells = dax::internal::numberOfCells(grid);

  dax::exec::WorkMapCell<dax::exec::CellVoxel> work(grid, cellIndex);
  dax::exec::FieldCoordinates pointCoord
      = dax::exec::internal::fieldCoordinatesBuild(grid);
  dax::exec::FieldPoint<dax::Scalar> inPointField(inPointScalars);
  dax::exec::FieldCell<dax::Vector3> outCellField(outCellVectors);

  for ( ; cellIndex < numCells; cellIndex += cellIncrement)
    {
    work.SetCellIndex(cellIndex);
    CellGradient(work, pointCoord, inPointField, outCellField);
    }
}

__global__ void ExecutePipeline2(
    dax::internal::StructureUniformGrid grid,
    dax::internal::DataArray<dax::Scalar> inPointScalars,
    dax::internal::DataArray<dax::Vector3> outCellVectors)
{
  dax::Id cellIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
  dax::Id cellIncrement = gridDim.x;
  dax::Id numCells = dax::internal::numberOfCells(grid);

  dax::Vector3 intermediateValue1;
  dax::internal::DataArray<dax::Vector3> intermediateArray1;

  dax::Vector3 intermediateValue2;
  dax::internal::DataArray<dax::Vector3> intermediateArray2;

  dax::Vector3 intermediateValue3;
  dax::internal::DataArray<dax::Vector3> intermediateArray3;

  dax::exec::WorkMapCell<dax::exec::CellVoxel> workCell(grid, cellIndex);
  dax::exec::WorkMapField<dax::exec::CellVoxel> workField(grid, cellIndex);
  dax::exec::FieldCoordinates pointCoord
      = dax::exec::internal::fieldCoordinatesBuild(grid);
  dax::exec::FieldPoint<dax::Scalar> inPointField(inPointScalars);
  dax::exec::FieldCell<dax::Vector3> outCellField(outCellVectors);

  for ( ; cellIndex < numCells; cellIndex += cellIncrement)
    {
    workCell.SetCellIndex(cellIndex);
    workField.SetIndex(cellIndex);

    intermediateArray1.SetPointer(&intermediateValue1, 1, cellIndex);
    dax::exec::FieldCell<dax::Vector3> intermediateField1(intermediateArray1);
    CellGradient(workCell, pointCoord, inPointField, intermediateField1);

    intermediateArray2.SetPointer(&intermediateValue2, 1, cellIndex);
    dax::exec::FieldCell<dax::Vector3> intermediateField2(intermediateArray2);
    Sine(workField, intermediateField1, intermediateField2);

    intermediateArray3.SetPointer(&intermediateValue3, 1, cellIndex);
    dax::exec::FieldCell<dax::Vector3> intermediateField3(intermediateArray3);
    Square(workField, intermediateField2, intermediateField3);

    Cosine(workField, intermediateField3, outCellField);
    }
}

__global__ void ExecutePipeline3(
    dax::internal::StructureUniformGrid grid,
    dax::internal::DataArray<dax::Scalar> outPointScalars)
{
  dax::Id pointIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
  dax::Id pointIncrement = gridDim.x;
  dax::Id numPoints = dax::internal::numberOfPoints(grid);

  dax::Scalar intermediateValue1;
  dax::internal::DataArray<dax::Scalar> intermediateArray1;

  dax::Scalar intermediateValue2;
  dax::internal::DataArray<dax::Scalar> intermediateArray2;

  dax::Scalar intermediateValue3;
  dax::internal::DataArray<dax::Scalar> intermediateArray3;

  dax::exec::WorkMapField<dax::exec::CellVoxel> work(grid, pointIndex);
  dax::exec::FieldCoordinates pointCoord
      = dax::exec::internal::fieldCoordinatesBuild(grid);
  dax::exec::FieldPoint<dax::Scalar> outField(outPointScalars);

  for ( ; pointIndex < numPoints; pointIndex += pointIncrement)
    {
    work.SetIndex(pointIndex);

    intermediateArray1.SetPointer(&intermediateValue1, 1, pointIndex);
    dax::exec::FieldPoint<dax::Scalar> intermediateField1(intermediateArray1);
    Elevation(work, pointCoord, intermediateField1);

    intermediateArray2.SetPointer(&intermediateValue2, 1, pointIndex);
    dax::exec::FieldPoint<dax::Scalar> intermediateField2(intermediateArray2);
    Sine(work, intermediateField1, intermediateField2);

    intermediateArray3.SetPointer(&intermediateValue3, 1, pointIndex);
    dax::exec::FieldPoint<dax::Scalar> intermediateField3(intermediateArray1);
    Square(work, intermediateField2, intermediateField3);

    Cosine(work, intermediateField3, outField);
    }
}

#include <iostream>
#include <vector>
using namespace std;

static void PrintCheckValues(const dax::internal::DataArray<dax::Vector3> &array)
{
  for (dax::Id index = 0; index < array.GetNumberOfEntries(); index++)
    {
    dax::Vector3 value = array.GetValue(index);
    if (index < 20)
      {
      cout << index << " : " << value.x << ", " << value.y << ", " << value.z
           << endl;
      }
    if (   (value.x < -1) || (value .x > 1)
        || (value.y < -1) || (value .y > 1)
        || (value.z < -1) || (value .z > 1) )
      {
      cout << index << " : " << value.x << ", " << value.y << ", " << value.z
           << endl;
      break;
      }
    }
}

static void PrintCheckValues(const dax::internal::DataArray<dax::Scalar> &array)
{
  for (dax::Id index = 0; index < array.GetNumberOfEntries(); index++)
    {
    dax::Scalar value = array.GetValue(index);
    if (index < 20)
      {
      cout << index << " : " << value << endl;
      }
    if ((value < -1) || (value > 1))
      {
      cout << "BAD VALUE " << index << " : " << value << endl;
      break;
      }
    }
}

static dax::internal::StructureUniformGrid CreateInputStructure(dax::Id dim)
{
  dax::internal::StructureUniformGrid grid;
  grid.Origin = dax::make_Vector3(0.0, 0.0, 0.0);
  grid.Spacing = dax::make_Vector3(1.0, 1.0, 1.0);
  grid.Extent.Min = dax::make_Id3(0, 0, 0);
  grid.Extent.Max = dax::make_Id3(dim-1, dim-1, dim-1);

  return grid;
}

static void RunCellGradient(dax::internal::StructureUniformGrid &grid,
                            dax::Id point_blockCount,
                            dax::Id point_threadsPerBlock,
                            dax::Id cell_blockCount,
                            dax::Id cell_threadsPerBlock,
                            double &upload_time,
                            double &execute_time,
                            double &download_time)
{
  dax::cuda::cont::internal::DeviceArray<dax::Scalar> elevationResult(
        dax::internal::numberOfPoints(grid));

  dax::cuda::cont::internal::DeviceArray<dax::Vector3> gradientResult(
        dax::internal::numberOfCells(grid));

  boost::timer timer;

  upload_time = 0.0;

  timer.restart();
  ExecuteElevation<<<point_blockCount, point_threadsPerBlock>>>(
        grid,
        elevationResult);
  ExecutePipeline1<<<cell_blockCount, cell_threadsPerBlock>>>(
        grid,
        elevationResult,
        gradientResult);
  if (cudaThreadSynchronize() != cudaSuccess)
    {
    abort();
    }
  execute_time = timer.elapsed();

  // Download the result, just to time it.
  dax::cont::HostArray<dax::Vector3> hostResult(numberOfCells(grid));
  timer.restart();
  hostResult = gradientResult;
  download_time = timer.elapsed();

  PrintCheckValues(hostResult);
}

static void RunCellGradientSinSqrCos(dax::internal::StructureUniformGrid &grid,
                                     dax::Id point_blockCount,
                                     dax::Id point_threadsPerBlock,
                                     dax::Id cell_blockCount,
                                     dax::Id cell_threadsPerBlock,
                                     double &upload_time,
                                     double &execute_time,
                                     double &download_time)
{
  dax::cuda::cont::internal::DeviceArray<dax::Scalar> elevationResult(
        dax::internal::numberOfPoints(grid));

  dax::cuda::cont::internal::DeviceArray<dax::Vector3> finalResult(
        dax::internal::numberOfCells(grid));

  boost::timer timer;

  upload_time = 0.0;

  timer.restart();
  ExecuteElevation<<<point_blockCount, point_threadsPerBlock>>>(
        grid,
        elevationResult);
  ExecutePipeline2<<<cell_blockCount, cell_threadsPerBlock>>>(
        grid,
        elevationResult,
        finalResult);
  if (cudaThreadSynchronize() != cudaSuccess)
    {
    abort();
    }
  execute_time = timer.elapsed();

  // Download the result, just to time it.
  dax::cont::HostArray<dax::Vector3> hostResult(numberOfCells(grid));
  timer.restart();
  hostResult = finalResult;
  download_time = timer.elapsed();

  PrintCheckValues(hostResult);
}

static void RunSinSqrCos(dax::internal::StructureUniformGrid &grid,
                         dax::Id point_blockCount,
                         dax::Id point_threadsPerBlock,
                         double &upload_time,
                         double &execute_time,
                         double &download_time)
{
  dax::cuda::cont::internal::DeviceArray<dax::Scalar> finalResult(
        dax::internal::numberOfPoints(grid));

  boost::timer timer;

  upload_time = 0.0;

  timer.restart();
  ExecutePipeline3<<<point_blockCount, point_threadsPerBlock>>>(
        grid,
        finalResult);
  if (cudaThreadSynchronize() != cudaSuccess)
    {
    abort();
    }
  execute_time = timer.elapsed();

  // Download the result, just to time it.
  dax::cont::HostArray<dax::Scalar> hostResult;

  timer.restart();
  hostResult = finalResult;
  download_time = timer.elapsed();

  PrintCheckValues(hostResult);
}


int main(int argc, char* argv[])
{
  dax::testing::ArgumentsParser parser;
  if (!parser.ParseArguments(argc, argv))
    {
    return 1;
    }

  const dax::Id MAX_SIZE = parser.GetProblemSize();
  // I think you mean threads per block, not warp size, which is different.
  // I don't think you can control the warp size.
  const dax::Id MAX_WARP_SIZE = parser.GetMaxWarpSize();
  // I think what you actually mean here is blocks per grid. I believe in CUDA
  // "grid size" is meant to be the total number of threads (threadsPerBlock *
  // blocksPerGrid).
  const dax::Id MAX_GRID_SIZE = parser.GetMaxGridSize();

  dax::Id numCells = (MAX_SIZE-1) * (MAX_SIZE-1) * (MAX_SIZE-1);
  dax::Id cell_threadsPerBlock = min(MAX_WARP_SIZE, numCells);
  dax::Id cell_blockCount = min(MAX_GRID_SIZE, numCells/cell_threadsPerBlock+1);
  cout << "Execute (Cell) blockCount="  << cell_blockCount
       << ", threadsPerBlock=" << cell_threadsPerBlock << endl;

  dax::Id numPoints = MAX_SIZE * MAX_SIZE * MAX_SIZE;
  dax::Id point_threadsPerBlock = min(MAX_WARP_SIZE, numPoints);
  dax::Id point_blockCount =
      min(MAX_GRID_SIZE, numPoints/point_threadsPerBlock+1);
  cout << "Execute (Point) blockCount="  << point_blockCount
       << ", threadsPerBlock=" << point_threadsPerBlock << endl;


  boost::timer timer;

  timer.restart();
  dax::internal::StructureUniformGrid grid = CreateInputStructure(MAX_SIZE);
  double init_time = timer.elapsed();

  double upload_time;
  double execute_time;
  double download_time;

  switch (parser.GetPipeline())
    {
  case dax::testing::ArgumentsParser::CELL_GRADIENT:
    cout << "Pipeline #1" << endl;
    RunCellGradient(grid,
                    point_blockCount, point_threadsPerBlock,
                    cell_blockCount, cell_threadsPerBlock,
                    upload_time,
                    execute_time,
                    download_time);
    break;
  case dax::testing::ArgumentsParser::CELL_GRADIENT_SINE_SQUARE_COS:
    cout << "Pipeline #2" << endl;
    RunCellGradientSinSqrCos(grid,
                             point_blockCount, point_threadsPerBlock,
                             cell_blockCount, cell_threadsPerBlock,
                             upload_time,
                             execute_time,
                             download_time);
    break;

  case dax::testing::ArgumentsParser::SINE_SQUARE_COS:
    cout << "Pipeline #3" << endl;
    RunSinSqrCos(grid,
                 point_blockCount, point_threadsPerBlock,
                 upload_time,
                 execute_time,
                 download_time);
    break;
    }


  cout << endl << endl << "Summary: -- " << MAX_SIZE << "^3 Dataset" << endl;
  cout << "Initialize: " << init_time << endl
       << "Upload: " << upload_time << endl
       << "Execute: " << execute_time << endl
       << "Download: " << download_time << endl;
  return 0;
}
