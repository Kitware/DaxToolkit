/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/TypeTraits.h>

#include <dax/internal/Testing.h>

namespace {

/// Compares some manual arithmetic through type traits to overloaded
/// arithmetic that should be tested separately in UnitTestTypes.
template <class T>
static void TestVectorType(const T &value)
{
  typedef typename dax::VectorTraits<T> Traits;

  {
  T result;
  const typename Traits::ValueType multiplier = 4;
  for (int i = 0; i < Traits::NUM_COMPONENTS; i++)
    {
    Traits::SetComponent(result, i, multiplier*Traits::GetComponent(value, i));
    }
  DAX_TEST_ASSERT(result == multiplier*value,
                  "Got bad result for scalar multiple");
  }

  {
  T result;
  const typename Traits::ValueType multiplier = 7;
  for (int i = 0; i < Traits::NUM_COMPONENTS; i++)
    {
    Traits::GetComponent(result, i)
        = multiplier * Traits::GetComponent(value, i);
    }
  DAX_TEST_ASSERT(result == multiplier*value,
                  "Got bad result for scalar multiple");
  }

  {
  typename Traits::ValueType result = 0;
  for (int i = 0; i < Traits::NUM_COMPONENTS; i++)
    {
    typename Traits::ValueType component
        = Traits::GetComponent(value, i);
    result += component * component;
    }
  DAX_TEST_ASSERT(result == dax::dot(value, value),
                  "Got bad result for dot product");
  }
}

static void CheckVectorComponentsTag(dax::VectorTraitsTagMultipleComponents)
{
  // If we are running here, everything is fine.
}

/// Checks to make sure that the HasMultipleComponents tag is actually for
/// multiple components. Should only be called for vector classes that actually
/// have multiple components.
///
template<class T>
static void TestVectorComponentsTag()
{
  // This will fail to compile if the tag is wrong
  // (i.e. not dax::VectorTraitsTagMultipleComponents)
  CheckVectorComponentsTag(
        typename dax::VectorTraits<T>::HasMultipleComponents());
}

static void CheckScalarComponentsTag(dax::VectorTraitsTagSingleComponent)
{
  // If we are running here, everything is fine.
}

/// Checks to make sure that the HasMultipleComponents tag is actually for a
/// single component. Should only be called for "vector" classes that actually
/// have only a single component (that is, are really scalars).
///
template<class T>
static void TestScalarComponentsTag()
{
  // This will fail to compile if the tag is wrong
  // (i.e. not dax::VectorTraitsTagSingleComponent)
  CheckScalarComponentsTag(
        typename dax::VectorTraits<T>::HasMultipleComponents());
}

void TestTypeTraits()
{
  std::cout << "Testing Id3" << std::endl;
  TestVectorType(dax::make_Id3(42, 54, 67));
  TestVectorComponentsTag<dax::Id3>();
  std::cout << "Testing Vector3" << std::endl;
  TestVectorType(dax::make_Vector3(42, 54, 67));
  TestVectorComponentsTag<dax::Vector3>();
  std::cout << "Testing Vector4" << std::endl;
  TestVectorType(dax::make_Vector4(42, 54, 67, 12));
  TestVectorComponentsTag<dax::Vector4>();
  std::cout << "Testing Id" << std::endl;
  TestVectorType(static_cast<dax::Id>(42));
  TestScalarComponentsTag<dax::Id>();
  std::cout << "Testing Scalar" << std::endl;
  TestVectorType(static_cast<dax::Scalar>(42));
  TestScalarComponentsTag<dax::Scalar>();
}

} // anonymous namespace

int UnitTestTypeTraits(int, char *[])
{
  return dax::internal::Testing::Run(TestTypeTraits);
}
