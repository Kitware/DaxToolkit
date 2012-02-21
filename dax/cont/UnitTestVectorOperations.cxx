/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/VectorOperations.h>

#include <dax/cont/internal/Testing.h>

namespace {

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

  VectorType squaredVector = dax::cont::VectorMap(value, Square<ValueType>);
  DAX_TEST_ASSERT(squaredVector == value*value,
                  "Got bad result for squaring vector components");

  ValueType magSquared = dax::cont::VectorReduce(squaredVector, Add<ValueType>);
  DAX_TEST_ASSERT(magSquared == dax::dot(value, value),
                  "Got bad result for summing vector components");

  {
  Summation<ValueType> sum;
  dax::cont::VectorForEach(squaredVector, sum);
  DAX_TEST_ASSERT(sum.Sum == magSquared,
                  "Got bad result for summing with VectorForEach");
  }

  {
  // Repeat the last test with a const reference.
  Summation<ValueType> sum;
  const VectorType &constSquaredVector = squaredVector;
  dax::cont::VectorForEach(constSquaredVector, sum);
  DAX_TEST_ASSERT(sum.Sum == magSquared,
                  "Got bad result for summing with VectorForEach");
  }

  {
  // Test the VectorFill function.
  ValueType fillValue = Traits::GetComponent(value, 0);
  VectorType fillVector;
  dax::cont::VectorFill(fillVector, fillValue);
  Summation<ValueType> sum;
  dax::cont::VectorForEach(fillVector, sum);
  DAX_TEST_ASSERT(sum.Sum == fillValue * Traits::NUM_COMPONENTS,
                  "Got bad result filling vector");
  }

  {
  // Test another form of the VectorFill function.
  ValueType fillValue = Traits::GetComponent(value, 0);
  VectorType fillVector = dax::cont::VectorFill<VectorType>(fillValue);
  Summation<ValueType> sum;
  dax::cont::VectorForEach(fillVector, sum);
  DAX_TEST_ASSERT(sum.Sum == fillValue * Traits::NUM_COMPONENTS,
                  "Got bad result filling vector");
  }
}

static void TestVectorTypes()
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

} // anonymous namespace

int UnitTestVectorOperations(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestVectorTypes);
}
