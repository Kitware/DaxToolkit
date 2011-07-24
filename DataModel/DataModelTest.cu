
#include "daxImageData.h"
#include "daxDataArrayIrregular.h"
#include "daxDataBridge.h"
#include "daxKernelArgument.h"
#include "DaxKernelArgument.h"

#include <math.h>
#include <thrust/device_vector.h>

__global__ void Execute(DaxKernelArgument argument, int *temp)
{
  *temp = argument.NumberOfArrays;
}

int main()
{
  daxImageDataPtr imageData(new daxImageData());
  imageData->SetExtent(0, 100, 0, 100, 0, 100);
  imageData->SetOrigin(0, 0, 0);
  imageData->SetSpacing(1, 1, 1);

  daxDataArrayScalarPtr point_scalars (new daxDataArrayScalar());
  point_scalars->SetName("Scalars");
  point_scalars->SetNumberOfTuples(imageData->GetNumberOfPoints());
  imageData->PointData.push_back(point_scalars);

  for (int x=0 ; x < 100; x ++)
    {
    for (int y=0 ; y < 100; y ++)
      {
      for (int z=0 ; z < 100; z ++)
        {
        point_scalars->Set(
          z * 100 * 100 + y * 100 + x, sqrt(x*x+y*y+z*z));
        }
      }
    }

  daxImageDataPtr imageData2(new daxImageData());
  imageData2->SetExtent(0, 100, 0, 100, 0, 100);
  imageData2->SetOrigin(0, 0, 0);
  imageData2->SetSpacing(1, 1, 1);

  daxDataArrayScalarPtr point_scalars2 (new daxDataArrayScalar());
  point_scalars2->SetName("Scalars");
  point_scalars2->SetNumberOfTuples(imageData->GetNumberOfCells());
  imageData2->CellData.push_back(point_scalars2);

  daxDataBridge bridge;
  bridge.AddInputData(imageData);
  bridge.AddOutputData(imageData2);
  daxKernelArgumentPtr arg = bridge.Upload();

  thrust::device_vector<int> temp(1);
  *temp.begin() = -1;

  Execute<<<1, 1>>>(arg->Get(), thrust::raw_pointer_cast(&temp[0]));
  cout << "Value: " << *temp.begin() << endl;
}
