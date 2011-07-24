/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include "DaxExecutionEnvironment.h"

#include "daxImageData.h"
#include "daxDataArrayIrregular.h"
#include "daxDataBridge.h"
#include "daxKernelArgument.h"
#include "DaxKernelArgument.h"

#include "PointDataToCellData.worklet"
#include "CellGradient.worklet"
#include "CellAverage.worklet"

__global__ void Execute(DaxKernelArgument argument)
{
  DaxWorkMapCell work(
    argument.Arrays[
      argument.Datasets[0].CellArrayIndex]);

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


#define MAX_SIZE 32

int main()
{
  daxImageDataPtr input = CreateInputDataSet(MAX_SIZE);
  daxImageDataPtr output = CreateOutputDataSet(MAX_SIZE);

  daxDataBridge bridge;
  bridge.AddInputData(input);
  bridge.AddOutputData(output);

  daxKernelArgumentPtr arg = bridge.Upload();
  Execute<<< (MAX_SIZE-1)*(MAX_SIZE-1), MAX_SIZE >>>(arg->Get());
  bridge.Download(arg);

  daxDataArrayVector3* array = dynamic_cast<
    daxDataArrayVector3*>( &(*output->CellData[0]) );
  for (size_t cc=0; cc < array->GetNumberOfTuples(); cc++)
    {
    DaxVector3 value = array->Get(cc);
    cout << cc << " : " << value.x << ", " << value.y << ", " << value.z << endl;
    if (cc == 20) break;
    }
  return 0;
}
