/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxArray.h"
#include "daxCellAverageModule.h"
#include "daxElevationModule.h"
#include "daxExecutive2.h"

#include <string.h>
#include <assert.h>

int main(int, char**)
{
  daxArrayPtr array2(new daxArray());
  daxArrayPtr array(new daxArray());
  array->Set(daxArray::ELEMENT_TYPE(), 12);
  array->Set(daxArray::DEP(), array2);
  array->Set(daxArray::REF(), array2);
  cout << array->Get(daxArray::ELEMENT_TYPE()) << endl;
  
  daxExecutive2Ptr executive(new daxExecutive2());
  daxModulePtr moduleA(new daxElevationModule());

  daxModulePtr moduleB(new daxCellAverageModule());

  executive->Connect(moduleA, "output", moduleB, "input_point");
  executive->PrintKernel();


  //// Create input-buffer.
  //daxImageDataPtr inputData(new daxImageData());
  //inputData->SetDimensions(100, 100, 100);
  //float offset = 0.0f;
  //for (int x=0; x <inputData->GetDimensions()[0]; x++)
  //  {
  //  for (int y=0; y < inputData->GetDimensions()[1]; y++)
  //    {
  //    for (int z=0; z < inputData->GetDimensions()[2]; z++)
  //      {
  //      inputData->GetDataPointer(x, y, z)[0] = offset++;
  //      }
  //    }
  //  }

  //daxImageDataPtr outputData(new daxImageData());
  //outputData->SetDimensions(100, 100, 100);
  //memset(outputData->GetData(), 0, outputData->GetDataSize(NULL));

  //executive->Execute(inputData.get(), outputData.get());

  //for (int x=0; x <inputData->GetDimensions()[0]; x++)
  //  {
  //  for (int y=0; y < inputData->GetDimensions()[1]; y++)
  //    {
  //    for (int z=0; z < inputData->GetDimensions()[2]; z++)
  //      {
  //      assert(inputData->GetDataPointer(x, y, z)[0] ==
  //        outputData->GetDataPointer(x, y, z)[0]);
  //      }
  //    }
  //  }

  return 0;
}
