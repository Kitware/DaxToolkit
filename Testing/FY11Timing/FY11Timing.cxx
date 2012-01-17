/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

#include "ArgumentsParser.h"

#include "FY11TimingConfig.h"

#include "PipelineDebug.h"

#ifdef DAX_ENABLE_OPENMP
#include "PipelineOpenMP.h"
#endif

#include <iostream>

namespace {

dax::cont::UniformGrid CreateInputStructure(dax::Id dim)
{
  dax::cont::UniformGrid grid;
  grid.SetOrigin(dax::make_Vector3(0.0, 0.0, 0.0));
  grid.SetSpacing(dax::make_Vector3(1.0, 1.0, 1.0));
  grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(dim-1, dim-1, dim-1));
  return grid;
}

}

int main(int argc, char* argv[])
  {
  dax::testing::ArgumentsParser parser;
  if (!parser.parseArguments(argc, argv))
    {
    return 1;
    }

  //init grid vars from parser
  const dax::Id MAX_SIZE = parser.problemSize();

  dax::cont::UniformGrid grid = CreateInputStructure(MAX_SIZE);

  int pipeline = parser.pipeline();
  std::cout << "Pipeline #" << pipeline << std::endl;

  switch (parser.device())
    {
    case dax::testing::ArgumentsParser::DEVICE_ALL:
      RunPipelineDebug(pipeline, grid);
      RunPipelineOpenMP(pipeline, grid);
      break;
    case dax::testing::ArgumentsParser::DEVICE_DEBUG:
      RunPipelineDebug(pipeline, grid);
      break;
    case dax::testing::ArgumentsParser::DEVICE_OPENMP:
#ifdef DAX_ENABLE_OPENMP
      RunPipelineOpenMP(pipeline, grid);
      break;
#else
      std::cout << "OpenMP device not available." << std::endl;
      return 1;
#endif
    case dax::testing::ArgumentsParser::DEVICE_CUDA:
      break;
    }

  return 0;
}
