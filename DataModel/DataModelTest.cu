
#include "daxImageData.h"
#include "daxDataArrayIrregular.h"
#include "daxDataBridge.h"

#include <math.h>

#include <thrust/device_vector.h>

struct Array
{
  void* RawData;
};


struct DataSet
{
  int foo;
  int bar;
  int asd;
  int num_arrays;
  Array* Arrays;
};

struct KernelArgument
{
  DataSet* DataSets;
  Array* Arrays;
  int DataSetCount;
  int ArrayCount;
};

__device__ int *foo;
__global__ void Execute(KernelArgument argument)
{

  int temp  = *foo;
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

  daxDataBridge bridge;
  bridge.AddInputData(imageData);

  //Execute<<<10, 10>>>( argument);
}
