/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

#include <dax/exec/math/Sign.h>

#include <dax/internal/Testing.h>

namespace {

template<typename VectorType>
void TestAbs(VectorType negative, VectorType positive)
{
  std::cout << "Testing absolute value "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;

  DAX_TEST_ASSERT(test_equal(dax::exec::math::Abs(negative), positive),
                  "Absolute value not flipping negative.");
  DAX_TEST_ASSERT(test_equal(dax::exec::math::Abs(positive), positive),
                  "Absolute value not leaving positive.");
}

void TestIsNegative(dax::Scalar negative, dax::Scalar positive)
{
  std::cout << "Testing IsNegative" << std::endl;

  DAX_TEST_ASSERT(dax::exec::math::IsNegative(negative),
                  "Did not detect negative.");
  DAX_TEST_ASSERT(!dax::exec::math::IsNegative(positive),
                  "Did not detect positive.");
  DAX_TEST_ASSERT(!dax::exec::math::IsNegative(dax::Scalar(0.0)),
                  "Did not detect non-negative.");
}

void TestCopySign(dax::Scalar negative, dax::Scalar positive)
{
  std::cout << "Testing CopySign" << std::endl;

  DAX_TEST_ASSERT(test_equal(-negative,
                             dax::exec::math::CopySign(negative, positive)),
                  "Did not copy positive sign.");
  DAX_TEST_ASSERT(test_equal(-positive,
                             dax::exec::math::CopySign(positive, negative)),
                  "Did not copy negative sign.");
  DAX_TEST_ASSERT(test_equal(-dax::exec::math::Abs(negative),
                             dax::exec::math::CopySign(negative, -1)),
                  "Did not copy negative sign.");
  DAX_TEST_ASSERT(test_equal(-dax::exec::math::Abs(positive),
                             dax::exec::math::CopySign(positive, -1)),
                  "Did not copy negative sign.");
  DAX_TEST_ASSERT(test_equal(dax::exec::math::Abs(negative),
                             dax::exec::math::CopySign(negative, 1)),
                  "Did not copy positive sign.");
  DAX_TEST_ASSERT(test_equal(dax::exec::math::Abs(positive),
                             dax::exec::math::CopySign(positive, 1)),
                  "Did not copy positive sign.");
  DAX_TEST_ASSERT(test_equal(dax::Scalar(-1),
                             dax::exec::math::CopySign(1, negative)),
                  "Did not copy negative sign.");
  DAX_TEST_ASSERT(test_equal(dax::Scalar(1),
                             dax::exec::math::CopySign(1, positive)),
                  "Did not copy positive sign.");
}


void TestSign()
{
  TestAbs<dax::Scalar>(-1.0, 1.0);
  TestAbs(dax::make_Vector3(0.0, -0.5, -3241.12),
          dax::make_Vector3(0.0,  0.5,  3241.12));
  TestAbs<dax::Id>(-1, 1);
  TestAbs(dax::make_Id3(0, -23, -652),
          dax::make_Id3(0,  23,  652));
  TestIsNegative(-2.3, 4.5);
  TestCopySign(-2.3, 4.5);
}

} // anonymous namespace

int UnitTestMathSign(int, char *[])
{
  return dax::internal::Testing::Run(TestSign);
}
