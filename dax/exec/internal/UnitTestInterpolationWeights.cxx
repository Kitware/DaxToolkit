/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/exec/internal/InterpolationWeights.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define TEST_FAIL(msg)                                  \
  {                                                     \
    std::stringstream error;                            \
    error << __FILE__ << ":" << __LINE__ << std::endl;  \
    error msg;                                          \
    throw error.str();                                  \
  }

static void TestInterpolationWeightsVoxel()
{
  std::cout << "In TestInterpolationWeightsVoxel" << std::endl;

  const dax::Vector3 cellVertexToParametricCoords[8] = {
    dax::make_Vector3(0, 0, 0),
    dax::make_Vector3(1, 0, 0),
    dax::make_Vector3(1, 1, 0),
    dax::make_Vector3(0, 1, 0),
    dax::make_Vector3(0, 0, 1),
    dax::make_Vector3(1, 0, 1),
    dax::make_Vector3(1, 1, 1),
    dax::make_Vector3(0, 1, 1)
  };

  // Check interpolation at each corner.
  for (dax::Id vertexIndex = 0; vertexIndex < 8; vertexIndex++)
    {
    dax::Vector3 pcoords = cellVertexToParametricCoords[vertexIndex];

    dax::Scalar weights[8];
    dax::exec::internal::interpolationWeightsVoxel(pcoords, weights);

    for (dax::Id weightIndex = 0; weightIndex < 8; weightIndex++)
      {
      if (weightIndex == vertexIndex)
        {
        if (weights[weightIndex] != 1.0)
          {
          TEST_FAIL(<< "Got bad interpolation weight");
          }
        }
      else
        {
        if (weights[weightIndex] != 0.0)
          {
          TEST_FAIL(<< "Got bad interpolation weight");
          }
        }
      }
    }

  // Check for interpolation at middle.
  dax::Scalar weights[8];
  dax::exec::internal::interpolationWeightsVoxel(dax::make_Vector3(0.5,0.5,0.5),
                                                 weights);
  for (dax::Id weightIndex = 0; weightIndex < 8; weightIndex++)
    {
    if (weights[weightIndex] != 0.125)
      {
      TEST_FAIL(<< "Got bad interpolation weight");
      }
    }
}

int UnitTestInterpolationWeights(int, char *[])
{
  try
    {
    TestInterpolationWeightsVoxel();
    }
  catch (std::string error)
    {
    std::cout
        << "Encountered error: " << std::endl
        << error << std::endl;
    return 1;
    }

  return 0;
}
