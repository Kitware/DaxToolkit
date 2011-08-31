/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include "ArgumentsParser.h"

// Includes for host code.
#include <dax/cont/DataArrayIrregular.h>
#include <dax/cont/ImageData.h>

#include <dax/cuda/internal/KernelArgument.h>
#include <dax/cuda/cont/internal/DataBridge.h>
#include <dax/cuda/cont/internal/KernelArgument.h>


// Includes for device code.
#include <dax/cuda/exec/ExecutionEnvironment.h>

#include <Worklets/CellAverage.worklet>
#include <Worklets/CellGradient.worklet>
#include <Worklets/Cosine.worklet>
#include <Worklets/Elevation.worklet>
#include <Worklets/PointDataToCellData.worklet>
#include <Worklets/Sine.worklet>
#include <Worklets/Square.worklet>

#include <boost/progress.hpp>


__global__ void ExecuteElevation(dax::cuda::internal::KernelArgument argument,
  unsigned int number_of_threads,
  unsigned int number_of_iterations)
{
  for (unsigned int cc=0; cc < number_of_iterations; cc++)
    {
    dax::exec::WorkMapField work(cc);
    dax::exec::FieldCoordinates in_points(
      argument.Arrays[
      argument.Datasets[0].PointCoordinatesIndex]);
    dax::exec::FieldPoint out_point_scalars (
      argument.Arrays[
      argument.Datasets[1].PointDataIndices[0]]);
    Elevation(work, in_points, out_point_scalars);
    }
}

__global__ void ExecutePipeline1(dax::cuda::internal::KernelArgument argument,
  unsigned int number_of_threads,
  unsigned int number_of_iterations)
{
  for (unsigned int cc=0; cc < number_of_iterations; cc++)
    {
    dax::exec::WorkMapCell work(
      argument.Arrays[
      argument.Datasets[0].CellArrayIndex], cc);
    if (work.GetItem() < number_of_threads)
      {
      dax::exec::FieldCoordinates in_points(
        argument.Arrays[
        argument.Datasets[0].PointCoordinatesIndex]);
      dax::exec::FieldPoint in_point_scalars (
        argument.Arrays[
        argument.Datasets[1].PointDataIndices[0]]);
      dax::exec::FieldCell out_cell_vectors(
        argument.Arrays[
        argument.Datasets[2].CellDataIndices[0]]);

      CellGradient(work, in_points,
        in_point_scalars, out_cell_vectors);
      }
    }
}

__global__ void ExecutePipeline2(dax::cuda::internal::KernelArgument argument,
  unsigned int number_of_threads,
  unsigned int number_of_iterations)
{
  for (unsigned int cc=0; cc < number_of_iterations; cc++)
    {
    dax::exec::WorkMapCell work(
      argument.Arrays[
      argument.Datasets[0].CellArrayIndex], cc);
    if (work.GetItem() < number_of_threads)
      {
      dax::exec::FieldCoordinates in_points(
        argument.Arrays[
        argument.Datasets[0].PointCoordinatesIndex]);
      dax::exec::FieldPoint in_point_scalars (
        argument.Arrays[
        argument.Datasets[1].PointDataIndices[0]]);
      dax::exec::FieldCell out_cell_vectors(
        argument.Arrays[
        argument.Datasets[2].CellDataIndices[0]]);

      CellGradient(work, in_points,
        in_point_scalars, out_cell_vectors);
      Sine(work, out_cell_vectors, out_cell_vectors);
      Square(work, out_cell_vectors, out_cell_vectors);
      Cosine(work, out_cell_vectors, out_cell_vectors);
      }
    }
}

__global__ void ExecutePipeline3(dax::cuda::internal::KernelArgument argument,
  unsigned int number_of_threads,
  unsigned int number_of_iterations)
{
  for (unsigned int cc=0; cc < number_of_iterations; cc++)
    {
    dax::exec::WorkMapField work(cc);
    dax::exec::FieldCoordinates in_points(
      argument.Arrays[
      argument.Datasets[0].PointCoordinatesIndex]);
    dax::exec::FieldPoint out_point_scalars (
      argument.Arrays[
      argument.Datasets[1].PointDataIndices[0]]);
     Elevation(work, in_points, out_point_scalars);
     SineScalar(work, out_point_scalars, out_point_scalars);
     SquareScalar(work, out_point_scalars, out_point_scalars);
     CosineScalar(work, out_point_scalars, out_point_scalars);
    }
}

#include <iostream>
using namespace std;

dax::cont::ImageDataPtr CreateInputDataSet(int dim)
{
  dax::cont::ImageDataPtr imageData(new dax::cont::ImageData());
  imageData->SetExtent(0, dim-1, 0, dim-1, 0, dim-1);
  imageData->SetOrigin(0, 0, 0);
  imageData->SetSpacing(1, 1, 1);
  return imageData;
}

dax::cont::ImageDataPtr CreateIntermediateDataset(int dim)
{
  dax::cont::ImageDataPtr imageData(new dax::cont::ImageData());
  imageData->SetExtent(0, dim-1, 0, dim-1, 0, dim-1);
  imageData->SetOrigin(0, 0, 0);
  imageData->SetSpacing(1, 1, 1);

  dax::cont::DataArrayScalarPtr point_scalars (new dax::cont::DataArrayScalar());
  point_scalars->SetName("ElevationScalars");
  point_scalars->SetNumberOfTuples(imageData->GetNumberOfPoints());
  imageData->PointData.push_back(point_scalars);
  return imageData;
}

dax::cont::ImageDataPtr CreateOutputDataSet(int dim)
{
  dax::cont::ImageDataPtr imageData(new dax::cont::ImageData());
  imageData->SetExtent(0, dim-1, 0, dim-1, 0, dim-1);
  imageData->SetOrigin(0, 0, 0);
  imageData->SetSpacing(1, 1, 1);

  dax::cont::DataArrayVector3Ptr cell_gradients (new dax::cont::DataArrayVector3());
  cell_gradients->SetName("CellScalars");
  cell_gradients->SetNumberOfTuples(imageData->GetNumberOfCells());
  imageData->CellData.push_back(cell_gradients);

  for (int x=0 ; x < dim-1; x ++)
    {
    for (int y=0 ; y < dim-1; y ++)
      {
      for (int z=0 ; z < dim-1; z ++)
        {
        cell_gradients->Set(
          z * (dim-1) * (dim-1) + y * (dim-1) + x, dax::make_Vector3(-1, 0, 0));
        }
      }
    }

  return imageData;
}


int main(int argc, char* argv[])
{
  dax::testing::ArgumentsParser parser;
  if (!parser.ParseArguments(argc, argv))
    {
    return 1;
    }

  const unsigned int MAX_SIZE = parser.GetProblemSize();
  const unsigned int MAX_WARP_SIZE = parser.GetMaxWarpSize();
  const unsigned int MAX_GRID_SIZE = parser.GetMaxGridSize();

  unsigned int cell_number_of_threads = (MAX_SIZE-1) * (MAX_SIZE-1) * (MAX_SIZE-1);
  unsigned int cell_threadCount = min(MAX_WARP_SIZE, cell_number_of_threads);
  unsigned int warpCount = (cell_number_of_threads / MAX_WARP_SIZE) +
    (((cell_number_of_threads % MAX_WARP_SIZE) == 0)? 0 : 1);
  unsigned int cell_blockCount = min(MAX_GRID_SIZE, max(1, warpCount));
  unsigned int cell_iterations = ceil(warpCount * 1.0 / MAX_GRID_SIZE);
  cout << "Execute (Cell) iterations="
    << cell_iterations << " : blockCount="  << cell_blockCount
    << ", threadCount=" << cell_threadCount << endl;

  unsigned int point_number_of_threads = MAX_SIZE * MAX_SIZE * MAX_SIZE;
  unsigned int point_threadCount = min(MAX_WARP_SIZE, point_number_of_threads);
  warpCount = (point_number_of_threads / MAX_WARP_SIZE) +
    (((point_number_of_threads % MAX_WARP_SIZE) == 0)? 0 : 1);
  unsigned int point_blockCount = min(MAX_GRID_SIZE, max(1, warpCount));
  unsigned int point_iterations = ceil(warpCount * 1.0 / MAX_GRID_SIZE);
  cout << "Execute (Point) iterations="
    << point_iterations << " : blockCount="  << point_blockCount
    << ", threadCount=" << point_threadCount << endl;


  boost::timer timer;

  timer.restart();
  dax::cont::ImageDataPtr input = CreateInputDataSet(MAX_SIZE);
  dax::cont::ImageDataPtr intermediate = CreateIntermediateDataset(MAX_SIZE);
  dax::cont::ImageDataPtr output = CreateOutputDataSet(MAX_SIZE);
  double init_time = timer.elapsed();

  dax::cuda::cont::internal::DataBridge bridge;
  bridge.AddInputData(input);
  if (parser.GetPipeline() == dax::testing::ArgumentsParser::SINE_SQUARE_COS)
    {
    bridge.AddOutputData(intermediate);
    }
  else
    {
    bridge.AddIntermediateData(intermediate);
    bridge.AddOutputData(output);
    }

  timer.restart();
  dax::cuda::cont::internal::KernelArgumentPtr arg = bridge.Upload();
  if (cudaThreadSynchronize() != cudaSuccess)
    {
    abort();
    }

  double upload_time = timer.elapsed();

  timer.restart();
  dax::cuda::internal::KernelArgument _arg= arg->Get();
  switch (parser.GetPipeline())
    {
  case dax::testing::ArgumentsParser::CELL_GRADIENT:
    cout << "Pipeline #1" << endl;
    ExecuteElevation<<<point_blockCount, point_threadCount>>>(
      _arg, point_number_of_threads, point_iterations);
    ExecutePipeline1<<<cell_blockCount, cell_threadCount>>>(
      _arg, cell_number_of_threads, cell_iterations);
    break;
  case dax::testing::ArgumentsParser::CELL_GRADIENT_SINE_SQUARE_COS:
    cout << "Pipeline #2" << endl;
    ExecuteElevation<<<point_blockCount, point_threadCount>>>(
      _arg, point_number_of_threads, point_iterations);
    ExecutePipeline2<<<cell_blockCount, cell_threadCount>>>(
      _arg, cell_number_of_threads, cell_iterations);
    break;

  case dax::testing::ArgumentsParser::SINE_SQUARE_COS:
    cout << "Pipeline #3" << endl;
    ExecutePipeline3<<<point_blockCount, point_threadCount>>>(
      _arg, point_number_of_threads, point_iterations);
    break;
    }
  if (cudaThreadSynchronize() != cudaSuccess)
    {
    abort();
    }

  double execute_time = timer.elapsed();
  timer.restart();
  bridge.Download(arg);
  if (cudaThreadSynchronize() != cudaSuccess)
    {
    abort();
    }
  double download_time = timer.elapsed();

  if (parser.GetPipeline() == dax::testing::ArgumentsParser::SINE_SQUARE_COS)
    {
    dax::cont::DataArrayScalar* array =
      dynamic_cast<dax::cont::DataArrayScalar*>(&(*intermediate->PointData[0]));
    for (size_t cc=0; cc < array->GetNumberOfTuples(); cc++)
      {
      dax::Scalar value = array->Get(cc);
      if (cc < 20)
        {
        cout << cc << " : " << value << endl;
        }
      if (value == -1 || value > 1) 
        {
        cout << cc << " : " << value << endl;
        break;
        }
      }
    }
  else
    {
    dax::cont::DataArrayVector3* array =
      dynamic_cast<dax::cont::DataArrayVector3*>(&(*output->CellData[0]));
    for (size_t cc=0; cc < array->GetNumberOfTuples(); cc++)
      {
      dax::Vector3 value = array->Get(cc);
      if (cc < 20)
        {
        cout << cc << " : " << value.x << ", " << value.y << ", " << value.z << endl;
        }
      if (value.x == -1 || value.x > 1) 
        {
        cout << cc << " : " << value.x << ", " << value.y << ", " << value.z << endl;
        break;
        }
      }
    }


  cout << endl << endl << "Summary: -- " << MAX_SIZE << "^3 Dataset" << endl;
  cout << "Initialize: " << init_time << endl
       << "Upload: " << upload_time << endl
       << "Execute: " << execute_time << endl
       << "Download: " << download_time << endl;
  return 0;
}
