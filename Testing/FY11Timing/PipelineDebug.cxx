
/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

#include <dax/cont/DeviceAdapterDebug.h>

#include "PipelineDebug.h"

#include "Pipeline.h"

void RunPipelineDebug(int pipeline, const dax::cont::UniformGrid &grid)
{
  switch (pipeline)
    {
    case 1:
      RunPipeline1(grid);
      break;
    case 2:
      RunPipeline2(grid);
      break;
    case 3:
      RunPipeline3(grid);
      break;
    default:
      std::cout << "Invalid pipeline selected." << std::endl;
      exit(1);
      break;
    }
}
