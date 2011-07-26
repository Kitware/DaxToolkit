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

#include "PointDataToCellData.worklet"
#include "CellGradient.worklet"
#include "CellAverage.worklet"

#include <boost/progress.hpp>


#define DEBUG_INDEX 0

__global__ void Execute(DaxKernelArgument argument,
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
        argument.Datasets[0].PointDataIndices[0]]);
      DaxFieldCell out_cell_vectors(
        argument.Arrays[
        argument.Datasets[1].CellDataIndices[0]]);

      CellGradient(work, in_points,
        in_point_scalars, out_cell_vectors);
#if DEBUG_INDEX
      out_cell_vectors.Set(work,
        make_DaxVector3(work.GetItem(), 0 ,0));
#endif
      }
    }
  //out_cell_vectors.Set(work,
  //  make_DaxVector3(work.GetItem(), 0, 0));


//  DaxWorkMapCell work(input_do.CellArray);
//  DaxFieldPoint in_point_scalars(input_do.PointData);
//  DaxFieldCell out_cell_scalars(output_p2c.CellData);
//
//  PointDataToCellData(work, in_point_scalars, out_cell_scalars);
//  //CellAverage(work, in_point_scalars, out_cell_scalars);
//
//  DaxFieldCoordinates in_points(input_do.PointCoordinates);
//  DaxFieldCell out_cell_scalars_cg(output_cg);
//  CellGradient(work, in_points, in_point_scalars, out_cell_scalars_cg);
}

#include <iostream>
using namespace std;

daxImageDataPtr CreateInputDataSet(int dim)
{
  daxImageDataPtr imageData(new daxImageData());
  imageData->SetExtent(0, dim-1, 0, dim-1, 0, dim-1);
  imageData->SetOrigin(0, 0, 0);
  imageData->SetSpacing(1, 1, 1);

  daxDataArrayScalarPtr point_scalars (new daxDataArrayScalar());
  point_scalars->SetName("Scalars");
  point_scalars->SetNumberOfTuples(imageData->GetNumberOfPoints());
  imageData->PointData.push_back(point_scalars);

  for (int x=0 ; x < dim; x ++)
    {
    for (int y=0 ; y < dim; y ++)
      {
      for (int z=0 ; z < dim; z ++)
        {
        point_scalars->Set(
          z * dim * dim + y * dim + x, sqrt(x*x+y*y+z*z));
        }
      }
    }

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
  daxImageDataPtr output = CreateOutputDataSet(MAX_SIZE);
  double init_time = timer.elapsed();

  daxDataBridge bridge;
  bridge.AddInputData(input);
  bridge.AddOutputData(output);

  timer.restart();
  daxKernelArgumentPtr arg = bridge.Upload();
  if (cudaThreadSynchronize() != cudaSuccess)
    {
    abort();
    }

  double upload_time = timer.elapsed();

  timer.restart();
  Execute<<<blockCount, threadCount>>>(arg->Get(), number_of_threads, iterations);
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
#if DEBUG_INDEX
    if (value.x != cc) 
#else
    if (value.x == -1 || value.x > 1) 
#endif
      {
      cout << cc << " : " << value.x << ", " << value.y << ", " << value.z << endl;
      break;
      }
    }
  cout << endl << endl << "Summary: -- " << endl;
  cout << "Initialize: " << init_time << endl
       << "Upload: " << upload_time << endl
       << "Execute: " << execute_time << endl
       << "Download: " << download_time << endl;
  return 0;
}
