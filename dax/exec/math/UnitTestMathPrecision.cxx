/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

#include <dax/exec/math/Precision.h>

#include <dax/internal/Testing.h>

namespace {

void TestNonFinites()
{
  std::cout << "Testing non-finites." << std::endl;

  dax::Scalar zero = 0.0;
  dax::Scalar finite = 1.0;
  dax::Scalar nan = dax::exec::math::Nan();
  dax::Scalar inf = dax::exec::math::Infinity();
  dax::Scalar neginf = dax::exec::math::NegativeInfinity();

  // General behavior.
  DAX_TEST_ASSERT(nan != nan, "Nan not equal itself.");
  DAX_TEST_ASSERT(!(nan >= zero), "Nan not greater or less.");
  DAX_TEST_ASSERT(!(nan <= zero), "Nan not greater or less.");
  DAX_TEST_ASSERT(!(nan >= finite), "Nan not greater or less.");
  DAX_TEST_ASSERT(!(nan <= finite), "Nan not greater or less.");

  DAX_TEST_ASSERT(neginf < inf, "Infinity big");
  DAX_TEST_ASSERT(zero < inf, "Infinity big");
  DAX_TEST_ASSERT(finite < inf, "Infinity big");
  DAX_TEST_ASSERT(zero > -inf, "-Infinity small");
  DAX_TEST_ASSERT(finite > -inf, "-Infinity small");

  DAX_TEST_ASSERT(!dax::exec::math::IsNan(zero), "Bad IsNan check.");
  DAX_TEST_ASSERT(!dax::exec::math::IsNan(finite), "Bad IsNan check.");
  DAX_TEST_ASSERT(dax::exec::math::IsNan(nan), "Bad IsNan check.");
  DAX_TEST_ASSERT(!dax::exec::math::IsNan(inf), "Bad IsNan check.");
  DAX_TEST_ASSERT(!dax::exec::math::IsNan(neginf), "Bad IsNan check.");

  DAX_TEST_ASSERT(!dax::exec::math::IsInf(zero), "Bad infinity check.");
  DAX_TEST_ASSERT(!dax::exec::math::IsInf(finite), "Bad infinity check.");
  DAX_TEST_ASSERT(!dax::exec::math::IsInf(nan), "Bad infinity check.");
  DAX_TEST_ASSERT(dax::exec::math::IsInf(inf), "Bad infinity check.");
  DAX_TEST_ASSERT(dax::exec::math::IsInf(neginf), "Bad infinity check.");

  DAX_TEST_ASSERT(dax::exec::math::IsFinite(zero), "Bad finite check.");
  DAX_TEST_ASSERT(dax::exec::math::IsFinite(finite), "Bad finite check.");
  DAX_TEST_ASSERT(!dax::exec::math::IsFinite(nan), "Bad finite check.");
  DAX_TEST_ASSERT(!dax::exec::math::IsFinite(inf), "Bad finite check.");
  DAX_TEST_ASSERT(!dax::exec::math::IsFinite(neginf), "Bad finite check.");
}

template<typename VectorType>
void TestFMod(VectorType numerator,
              VectorType denominator,
              VectorType remainder)
{
  std::cout << "Testing FMod "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;

  VectorType computed = dax::exec::math::FMod(numerator, denominator);
  DAX_TEST_ASSERT(test_equal(computed, remainder), "Bad remainder");
}

template<typename VectorType>
void TestRemainder(VectorType numerator,
                   VectorType denominator,
                   VectorType remainder,
                   VectorType quotient)
{
  std::cout << "Testing Remainder "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;

  VectorType computedRemainder
      = dax::exec::math::Remainder(numerator, denominator);
  DAX_TEST_ASSERT(test_equal(computedRemainder, remainder), "Bad remainder");

  dax::exec::VectorFill(computedRemainder, dax::Scalar(0.0));
  VectorType computedQuotient;

  computedRemainder = dax::exec::math::RemainderQuotient(numerator,
                                                         denominator,
                                                         computedQuotient);
  DAX_TEST_ASSERT(test_equal(computedRemainder, remainder), "Bad remainder");
  DAX_TEST_ASSERT(test_equal(computedQuotient, quotient), "Bad quotient");

  typedef dax::VectorTraits<VectorType> Traits;
  int iQuotient;
  dax::Scalar sRemainder;
  sRemainder
      = dax::exec::math::RemainderQuotient(Traits::GetComponent(numerator, 0),
                                           Traits::GetComponent(denominator, 0),
                                           iQuotient);
  DAX_TEST_ASSERT(test_equal(sRemainder, Traits::GetComponent(remainder, 0)),
                  "Bad remainder");
  DAX_TEST_ASSERT(test_equal(dax::Scalar(iQuotient),
                             Traits::GetComponent(quotient, 0)),
                  "Bad quotient");
}

template<typename VectorType>
void TestModF(VectorType x, VectorType integral, VectorType fractional)
{
  std::cout << "Testing ModF "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;

  VectorType computedIntegral;
  VectorType computedFractional;

  computedFractional = dax::exec::math::ModF(x, computedIntegral);

  DAX_TEST_ASSERT(test_equal(computedIntegral, integral), "Bad integral");
  DAX_TEST_ASSERT(test_equal(computedFractional, fractional), "Bad fractional");
}

template<typename VectorType>
void TestRound(VectorType x,
               VectorType xFloor,
               VectorType xCeil,
               VectorType xRound)
{
  std::cout << "Testing Round "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;

  DAX_TEST_ASSERT(test_equal(dax::exec::math::Floor(x), xFloor), "Bad floor");
  DAX_TEST_ASSERT(test_equal(dax::exec::math::Ceil(x), xCeil), "Bad ceil");
  DAX_TEST_ASSERT(test_equal(dax::exec::math::Round(x), xRound), "Bad round");
}


void TestPrecision()
{
  TestNonFinites();
  TestFMod<dax::Scalar>(6.5, 2.3, 1.9);
  TestFMod(dax::make_Vector3(3.8, 5.8, 9.3),
           dax::make_Vector3(3.0, 1.6, 3.1),
           dax::make_Vector3(0.8, 1.0, 0.0));
  TestRemainder<dax::Scalar>(6.5, 2.3, -0.4, 3.0);
  TestRemainder(dax::make_Vector4(3.8, 5.8, 9.3, 77.0),
                dax::make_Vector4(3.0, 1.6, 3.1, 19.0),
                dax::make_Vector4(0.8,-0.6, 0.0, 1.0),
                dax::make_Vector4(1.0, 4.0, 3.0, 4.0));
  TestModF<dax::Scalar>(1.9, 1.0, 0.9);
  TestModF(dax::make_Vector3(4.6, 0.1, 73.4),
           dax::make_Vector3(4.0, 0.0, 73.0),
           dax::make_Vector3(0.6, 0.1, 0.4));
  TestRound<dax::Scalar>(1.9, 1.0, 2.0, 2.0);
  TestRound(dax::make_Vector4(4.6, 0.1, 73.4, 55.0),
            dax::make_Vector4(4.0, 0.0, 73.0, 55.0),
            dax::make_Vector4(5.0, 1.0, 74.0, 55.0),
            dax::make_Vector4(5.0, 0.0, 73.0, 55.0));
}

} // anonymous namespace

int UnitTestMathPrecision(int, char *[])
{
  return dax::internal::Testing::Run(TestPrecision);
}
