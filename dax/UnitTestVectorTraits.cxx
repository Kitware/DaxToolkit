/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/VectorTraits.h>

#include <dax/internal/Testing.h>

namespace {

static void CompareDimensionalityTags(dax::TypeTraitsScalarTag,
                                      dax::VectorTraitsTagSingleComponent)
{
  // If we are here, everything is fine.
}
static void CompareDimensionalityTags(dax::TypeTraitsVectorTag,
                                      dax::VectorTraitsTagMultipleComponents)
{
  // If we are here, everything is fine.
}

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

  // This will fail to compile if the tags are wrong.
  CompareDimensionalityTags(
        typename dax::TypeTraits<T>::DimensionalityTag(),
        typename dax::VectorTraits<T>::HasMultipleComponents());
}

static const dax::Id MAX_VECTOR_SIZE = 4;
static const dax::Id VectorInit[MAX_VECTOR_SIZE] = { 42, 54, 67, 12 };

struct TestVectorTypeFunctor
{
  template <typename T> void operator()(const T&) const {
    typedef dax::VectorTraits<T> Traits;
    DAX_TEST_ASSERT(Traits::NUM_COMPONENTS <= MAX_VECTOR_SIZE,
                    "Need to update test for larger vectors.");
    T vector;
    for (int index = 0; index < Traits::NUM_COMPONENTS; index++)
      {
      Traits::SetComponent(vector, index, VectorInit[index]);
      }
    TestVectorType(vector);
  }
};

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

void TestVectorTraits()
{
  dax::internal::Testing::TryAllTypes(TestVectorTypeFunctor());

  TestVectorComponentsTag<dax::Id3>();
  TestVectorComponentsTag<dax::Vector3>();
  TestVectorComponentsTag<dax::Vector4>();
  TestScalarComponentsTag<dax::Id>();
  TestScalarComponentsTag<dax::Scalar>();
}

} // anonymous namespace

int UnitTestVectorTraits(int, char *[])
{
  return dax::internal::Testing::Run(TestVectorTraits);
}
