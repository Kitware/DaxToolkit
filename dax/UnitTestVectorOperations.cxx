/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/VectorOperations.h>

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

/// Simple functions to be used in conjunction with the vector operations.
///
template <class T>
static T Square(T x) { return x*x; }
template <class T>
static T Add(T x, T y) { return x + y; }

template<class T>
struct Summation {
  T Sum;
  Summation() : Sum(0) { }
  void operator()(T x) { Sum += x; }
};

/// Compares operations through generic vector operations with some other
/// overloaded operations that should be tested separately in UnitTestTypes.
///
template <class VectorType>
static void TestVectorType(const VectorType &value)
{
  typedef typename dax::VectorTraits<VectorType> Traits;
  typedef typename Traits::ValueType ValueType;

  VectorType squaredVector = dax::VectorMap(value, Square<ValueType>);
  if (squaredVector != value*value)
    {
    TEST_FAIL(<< "Got bad result for squaring vector components");
    }

  ValueType magSquared = dax::VectorReduce(squaredVector, Add<ValueType>);
  if (magSquared != dax::dot(value, value))
    {
    TEST_FAIL(<< "Got bad result for summing vector components");
    }

  {
  Summation<ValueType> sum;
  dax::VectorForEach(squaredVector, sum);
  if (sum.Sum != magSquared)
    {
    TEST_FAIL(<< "Got bad result for summing with VectorForEach");
    }
  }

  {
  // Repeat the last test with a const reference.
  Summation<ValueType> sum;
  const VectorType &constSquaredVector = squaredVector;
  dax::VectorForEach(constSquaredVector, sum);
  if (sum.Sum != magSquared)
    {
    TEST_FAIL(<< "Got bad result for summing with VectorForEach");
    }
  }
}

int UnitTestVectorOperations(int, char *[])
{
  try
    {
    std::cout << "Testing Id3" << std::endl;
    TestVectorType(dax::make_Id3(42, 54, 67));
    std::cout << "Testing Vector3" << std::endl;
    TestVectorType(dax::make_Vector3(42, 54, 67));
    std::cout << "Testing Vector4" << std::endl;
    TestVectorType(dax::make_Vector4(42, 54, 67, 12));
    std::cout << "Testing Id" << std::endl;
    TestVectorType(static_cast<dax::Id>(42));
    std::cout << "Testing Scalar" << std::endl;
    TestVectorType(static_cast<dax::Scalar>(42));
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
