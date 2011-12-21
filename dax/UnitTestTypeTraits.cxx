/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/TypeTraits.h>

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
  if (result != multiplier*value)
    {
    TEST_FAIL(<< "Got bad result for scalar multiple");
    }
  }

  {
  T result;
  const typename Traits::ValueType multiplier = 7;
  for (int i = 0; i < Traits::NUM_COMPONENTS; i++)
    {
    Traits::GetComponent(result, i)
        = multiplier * Traits::GetComponent(value, i);
    }
  if (result != multiplier*value)
    {
    TEST_FAIL(<< "Got bad result for scalar multiple");
    }
  }

  {
  typename Traits::ValueType result = 0;
  for (int i = 0; i < Traits::NUM_COMPONENTS; i++)
    {
    typename Traits::ValueType component
        = Traits::GetComponent(value, i);
    result += component * component;
    }
  if (result != dax::dot(value, value))
    {
    TEST_FAIL(<< "Got bad result for dot product");
    }
  }
}

int UnitTestTypeTraits(int, char *[])
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
