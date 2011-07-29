/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include "DaxExecutionEnvironment.h"

#include "DaxArgumentsParser.h"
#include "daxDataArrayIrregular.h"
#include "daxDataBridge.h"
#include "daxImageData.h"
#include "daxKernelArgument.h"
#include "DaxKernelArgument.h"

#include "CellAverage.worklet"
#include "CellGradient.worklet"
#include "Cosine.worklet"
#include "Elevation.worklet"
#include "PointDataToCellData.worklet"
#include "Sine.worklet"
#include "Square.worklet"

#include <boost/progress.hpp>


__global__ void ExecuteElevation(DaxKernelArgument argument,
  unsigned int number_of_threads,
  unsigned int number_of_iterations)
{
  for (unsigned int cc=0; cc < number_of_iterations; cc++)
    {
    DaxWorkMapField work(cc);
    DaxFieldCoordinates in_points(
      argument.Arrays[
      argument.Datasets[0].PointCoordinatesIndex]);
    DaxFieldPoint out_point_scalars (
      argument.Arrays[
      argument.Datasets[1].PointDataIndices[0]]);
    Elevation(work, in_points, out_point_scalars);
    }
}

__global__ void ExecutePipeline1(DaxKernelArgument argument,
  unsigned int number_of_threads,
  unsigned int number_of_iterations)
{
  for (unsigned int cc=0; cc < number_of_iterations; cc++)
    {
    DaxWorkMapCell work(
      argument.Arrays[
      argument.Datasets[0].CellArrayIndex], cc);
    if (work.GetItem() < number_of_threads)
      {
      DaxFieldCoordinates in_points(
        argument.Arrays[
        argument.Datasets[0].PointCoordinatesIndex]);
      DaxFieldPoint in_point_scalars (
        argument.Arrays[
        argument.Datasets[1].PointDataIndices[0]]);
      DaxFieldCell out_cell_vectors(
        argument.Arrays[
        argument.Datasets[2].CellDataIndices[0]]);

      CellGradient(work, in_points,
        in_point_scalars, out_cell_vectors);
      }
    }
}

__global__ void ExecutePipeline2(DaxKernelArgument argument,
  unsigned int number_of_threads,
  unsigned int number_of_iterations)
{
  for (unsigned int cc=0; cc < number_of_iterations; cc++)
    {
    DaxWorkMapCell work(
      argument.Arrays[
      argument.Datasets[0].CellArrayIndex], cc);
    if (work.GetItem() < number_of_threads)
      {
      DaxFieldCoordinates in_points(
        argument.Arrays[
        argument.Datasets[0].PointCoordinatesIndex]);
      DaxFieldPoint in_point_scalars (
        argument.Arrays[
        argument.Datasets[1].PointDataIndices[0]]);
      DaxFieldCell out_cell_vectors(
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

#include <iostream>
using namespace std;

daxImageDataPtr CreateInputDataSet(int dim)
{
  daxImageDataPtr imageData(new daxImageData());
  imageData->SetExtent(0, dim-1, 0, dim-1, 0, dim-1);
  imageData->SetOrigin(0, 0, 0);
  imageData->SetSpacing(1, 1, 1);
  return imageData;
}

daxImageDataPtr CreateIntermediateDataset(int dim)
{
  daxImageDataPtr imageData(new daxImageData());
  imageData->SetExtent(0, dim-1, 0, dim-1, 0, dim-1);
  imageData->SetOrigin(0, 0, 0);
  imageData->SetSpacing(1, 1, 1);

  daxDataArrayScalarPtr point_scalars (new daxDataArrayScalar());
  point_scalars->SetName("ElevationScalars");
  point_scalars->SetNumberOfTuples(imageData->GetNumberOfPoints());
  imageData->PointData.push_back(point_scalars);
  return imageData;
}

daxImageDataPtr CreateOutputDataSet(int dim)
{
  daxImageDataPtr imageData(new daxImageData());
  imageData->SetExtent(0, dim-1, 0, dim-1, 0, dim-1);
  imageData->SetOrigin(0, 0, 0);
  imageData->SetSpacing(1, 1, 1);

  daxDataArrayVector3Ptr cell_gradients (new daxDataArrayVector3());
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
          z * (dim-1) * (dim-1) + y * (dim-1) + x, make_DaxVector3(-1, 0, 0));
        }
      }
    }

  return imageData;
}



int main(int argc, char* argv[])
{
  DaxArgumentsParser parser;
  if (!parser.ParseArguments(argc, argv))
    {
    return 1;
    }

  const unsigned int MAX_SIZE = parser.GetProblemSize();
  const unsigned int MAX_WARP_SIZE = parser.GetMaxWarpSize();
  const unsigned int MAX_GRID_SIZE = parser.GetMaxGridSize();

  unsigned int number_of_threads = (MAX_SIZE-1) * (MAX_SIZE-1) * (MAX_SIZE-1);
  unsigned int threadCount = min(MAX_WARP_SIZE, number_of_threads);
  unsigned int warpCount = (number_of_threads / MAX_WARP_SIZE) +
    (((number_of_threads % MAX_WARP_SIZE) == 0)? 0 : 1);
  unsigned int blockCount = min(MAX_GRID_SIZE, max(1, warpCount));
  unsigned int iterations = ceil(warpCount * 1.0 / MAX_GRID_SIZE);
  cout << "Execute iterations="
    << iterations << " : blockCount="  << blockCount
    << ", threadCount=" << threadCount << endl;

  boost::timer timer;

  timer.restart();
  daxImageDataPtr input = CreateInputDataSet(MAX_SIZE);
  daxImageDataPtr intermediate = CreateIntermediateDataset(MAX_SIZE);
  daxImageDataPtr output = CreateOutputDataSet(MAX_SIZE);
  double init_time = timer.elapsed();

  daxDataBridge bridge;
  bridge.AddInputData(input);
  bridge.AddIntermediateData(intermediate);
  bridge.AddOutputData(output);

  timer.restart();
  daxKernelArgumentPtr arg = bridge.Upload();
  if (cudaThreadSynchronize() != cudaSuccess)
    {
    abort();
    }

  double upload_time = timer.elapsed();

  timer.restart();
  ExecuteElevation<<<blockCount, threadCount>>>(arg->Get(), number_of_threads, iterations);
  if (parser.GetPipeline() == DaxArgumentsParser::CELL_GRADIENT)
    {
    cout << "Pipeline #1" << endl;
    ExecutePipeline1<<<blockCount, threadCount>>>(arg->Get(), number_of_threads, iterations);
    }
  else
    {
    cout << "Pipeline #2" << endl;
    ExecutePipeline2<<<blockCount, threadCount>>>(arg->Get(), number_of_threads, iterations);
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

  daxDataArrayVector3* array = dynamic_cast<
    daxDataArrayVector3*>( &(*output->CellData[0]) );
  for (size_t cc=0; cc < array->GetNumberOfTuples(); cc++)
    {
    DaxVector3 value = array->Get(cc);
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
  cout << endl << endl << "Summary: -- " << MAX_SIZE << "^3 Dataset" << endl;
  cout << "Initialize: " << init_time << endl
       << "Upload: " << upload_time << endl
       << "Execute: " << execute_time << endl
       << "Download: " << download_time << endl;
  return 0;
}
