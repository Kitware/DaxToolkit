/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

#include <dax/exec/math/Compare.h>

#include <dax/internal/Testing.h>

namespace {

template<typename VectorType>
void TestMinMax(VectorType x, VectorType y)
{
  typedef dax::VectorTraits<VectorType> Traits;
  typedef typename Traits::ValueType ValueType;
  const dax::Id NUM_COMPONENTS = Traits::NUM_COMPONENTS;

  std::cout << "Testing Min and Max: " << NUM_COMPONENTS << " components"
            << std::endl;

  VectorType min = dax::exec::math::Min(x, y);
  VectorType max = dax::exec::math::Max(x, y);

  for (dax::Id index = 0; index < NUM_COMPONENTS; index++)
    {
    ValueType x_index = Traits::GetComponent(x, index);
    ValueType y_index = Traits::GetComponent(y, index);
    ValueType min_index = Traits::GetComponent(min, index);
    ValueType max_index = Traits::GetComponent(max, index);
    if (x_index < y_index)
      {
      DAX_TEST_ASSERT(x_index == min_index, "Got wrong min.");
      DAX_TEST_ASSERT(y_index == max_index, "Got wrong max.");
      }
    else
      {
      DAX_TEST_ASSERT(x_index == max_index, "Got wrong max.");
      DAX_TEST_ASSERT(y_index == min_index, "Got wrong min.");
      }
    }
}


void TestCompare()
{
  TestMinMax<dax::Scalar>(0.0, 1.0);
  TestMinMax<dax::Scalar>(0.0, -1.0);
  TestMinMax<dax::Id>(0, 1);
  TestMinMax<dax::Id>(0, -1);
  TestMinMax(dax::make_Id3(-4, -1, 2),
             dax::make_Id3(7, -6, 5));
  TestMinMax(dax::make_Vector3(-4, -1, 2),
             dax::make_Vector3(7, -6, 5));
  TestMinMax(dax::make_Vector4(-4, -1, 2, 0.0),
             dax::make_Vector4(7, -6, 5, -0.001));
}

} // anonymous namespace

int UnitTestMathCompare(int, char *[])
{
  return dax::internal::Testing::Run(TestCompare);
}
