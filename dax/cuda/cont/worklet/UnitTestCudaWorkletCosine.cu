/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cuda/cont/worklet/Cosine.h>

#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <dax/TypeTraits.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>

#include <vector>

namespace {

const dax::Id DIM = 64;
const dax::Scalar TOLERANCE = 0.0001;

#define test_assert(condition, message) \
  test_assert_impl(condition, message, __FILE__, __LINE__);

static inline void test_assert_impl(bool condition,
                                    const std::string& message,
                                    const char *file,
                                    int line)
{
  if(!condition)
    {
    std::stringstream error;
    error << file << ":" << line << std::endl;
    error << message << std::endl;
    throw error.str();
    }
}

template<typename VectorType>
static inline bool test_equal(VectorType vector1, VectorType vector2)
{
  typedef typename dax::VectorTraits<VectorType> Traits;
  for (int component = 0; component < Traits::NUM_COMPONENTS; component++)
    {
    dax::Scalar value1 = Traits::GetComponent(vector1, component);
    dax::Scalar value2 = Traits::GetComponent(vector2, component);
    if ((fabs(value1) < 2*TOLERANCE) && (fabs(value2) < 2*TOLERANCE))
      {
      continue;
      }
    dax::Scalar ratio = value1/value2;
    if ((ratio < 1.0 - TOLERANCE) || (ratio > 1.0 + TOLERANCE))
      {
      return false;
      }
    }
  return true;
}

//-----------------------------------------------------------------------------
static void TestCosine()
{
  dax::cont::UniformGrid grid;
  grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(DIM-1, DIM-1, DIM-1));

  dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);

  std::vector<dax::Scalar> field(grid.GetNumberOfPoints());
  for (dax::Id pointIndex = 0;
       pointIndex < grid.GetNumberOfPoints();
       pointIndex++)
    {
    field[pointIndex]
        = dax::dot(grid.GetPointCoordinates(pointIndex), trueGradient);
    }
  dax::cont::ArrayHandle<dax::Scalar> fieldHandle(field.begin(), field.end());

  std::vector<dax::Scalar> cosine(grid.GetNumberOfPoints());
  dax::cont::ArrayHandle<dax::Scalar> cosineHandle(cosine.begin(),
                                                   cosine.end());

  std::cout << "Running Cosine worklet" << std::endl;
  dax::cuda::cont::worklet::Cosine(grid, fieldHandle, cosineHandle);

  std::cout << "Checking result" << std::endl;
  for (dax::Id pointIndex = 0;
       pointIndex < grid.GetNumberOfPoints();
       pointIndex++)
    {
    dax::Scalar cosineValue = cosine[pointIndex];
    dax::Scalar cosineTrue = cosf(field[pointIndex]);
    test_assert(test_equal(cosineValue, cosineTrue),
                "Got bad gradient");
    }
}

} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestCudaWorkletCosine(int, char *[])
{
  try
    {
    TestCosine();
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
