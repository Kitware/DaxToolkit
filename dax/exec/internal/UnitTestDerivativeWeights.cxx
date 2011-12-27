/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/exec/internal/DerivativeWeights.h>

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

static void AssertDerivativeWeight(bool check)
{
  if (!check)
    {
    TEST_FAIL(<< "Got bad derivative weight");
    }
}

static void TestWeightOnVertex(dax::Vector3 weight,
                               dax::Vector3 derivativePCoord,
                               dax::Vector3 vertexPCoord)
{
  dax::Vector3 signs = 2.0*vertexPCoord - dax::make_Vector3(1.0, 1.0, 1.0);

  if (vertexPCoord == derivativePCoord)
    {
    AssertDerivativeWeight(weight == signs);
    }
  else if (   (vertexPCoord[0] != derivativePCoord[0])
           && (vertexPCoord[1] == derivativePCoord[1])
           && (vertexPCoord[2] == derivativePCoord[2]) )
    {
    AssertDerivativeWeight(weight == signs*dax::make_Vector3(1.0, 0.0, 0.0));
    }
  else if (   (vertexPCoord[0] == derivativePCoord[0])
           && (vertexPCoord[1] != derivativePCoord[1])
           && (vertexPCoord[2] == derivativePCoord[2]) )
    {
    AssertDerivativeWeight(weight == signs*dax::make_Vector3(0.0, 1.0, 0.0));
    }
  else if (   (vertexPCoord[0] == derivativePCoord[0])
           && (vertexPCoord[1] == derivativePCoord[1])
           && (vertexPCoord[2] != derivativePCoord[2]) )
    {
    AssertDerivativeWeight(weight == signs*dax::make_Vector3(0.0, 0.0, 1.0));
    }
  else
    {
    AssertDerivativeWeight(weight == dax::make_Vector3(0.0, 0.0, 0.0));
    }
}

static void TestWeightInMiddle(dax::Scalar weight,
                               dax::Scalar vertexPCoord)
{
  if (vertexPCoord < 0.5)
    {
    if (weight != -0.25)
      {
      TEST_FAIL(<< "Got bad derivative weight");
      }
    }
  else
    {
    if (weight != 0.25)
      {
      TEST_FAIL(<< "Got bad derivative weight");
      }
    }
}

static void TestDerivativeWeightsVoxel()
{
  std::cout << "In TestDerivativeWeightsVoxel" << std::endl;

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

  // Check Derivative at each corner.
  for (dax::Id vertexIndex = 0; vertexIndex < 8; vertexIndex++)
    {
    dax::Vector3 pcoords = cellVertexToParametricCoords[vertexIndex];

    dax::Vector3 weights[8];
    dax::exec::internal::derivativeWeightsVoxel(pcoords, weights);

    for (dax::Id weightIndex = 0; weightIndex < 8; weightIndex++)
      {
      dax::Vector3 vertexPCoords = cellVertexToParametricCoords[weightIndex];

      TestWeightOnVertex(weights[weightIndex], pcoords, vertexPCoords);
      }
    }

  // Check for Derivative at middle.
  dax::Vector3 weights[8];
  dax::exec::internal::derivativeWeightsVoxel(dax::make_Vector3(0.5,0.5,0.5),
                                              weights);
  for (dax::Id weightIndex = 0; weightIndex < 8; weightIndex++)
    {
    dax::Vector3 vertexPCoords = cellVertexToParametricCoords[weightIndex];

    TestWeightInMiddle(weights[weightIndex][0], vertexPCoords[0]);
    TestWeightInMiddle(weights[weightIndex][1], vertexPCoords[1]);
    TestWeightInMiddle(weights[weightIndex][2], vertexPCoords[2]);
    }
}

int UnitTestDerivativeWeights(int, char *[])
{
  try
    {
    TestDerivativeWeightsVoxel();
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
