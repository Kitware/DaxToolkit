/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

#include <dax/exec/math/Trig.h>

#include <dax/exec/math/Exp.h>

#include <dax/internal/Testing.h>

namespace {

void TestPi()
{
  std::cout << "Testing Pi" << std::endl;
  DAX_TEST_ASSERT(test_equal(dax::exec::math::Pi(), dax::Scalar(3.14159265)),
                  "Pi not correct.");
}

void TestArcTan2()
{
  std::cout << "Testing arc tan 2" << std::endl;

  DAX_TEST_ASSERT(test_equal(dax::exec::math::ATan2(0.0, 1.0),
                             dax::Scalar(0.0)),
                  "ATan2 x+ axis.");
  DAX_TEST_ASSERT(test_equal(dax::exec::math::ATan2(1.0, 0.0),
                             dax::Scalar(0.5*dax::exec::math::Pi())),
                  "ATan2 y+ axis.");
  DAX_TEST_ASSERT(test_equal(dax::exec::math::ATan2(-1.0, 0.0),
                             dax::Scalar(-0.5*dax::exec::math::Pi())),
                  "ATan2 y- axis.");

  DAX_TEST_ASSERT(test_equal(dax::exec::math::ATan2(1.0, 1.0),
                             dax::Scalar(0.25*dax::exec::math::Pi())),
                  "ATan2 Quadrant 1");
  DAX_TEST_ASSERT(test_equal(dax::exec::math::ATan2(1.0, -1.0),
                             dax::Scalar(0.75*dax::exec::math::Pi())),
                  "ATan2 Quadrant 2");
  DAX_TEST_ASSERT(test_equal(dax::exec::math::ATan2(-1.0, -1.0),
                             dax::Scalar(-0.75*dax::exec::math::Pi())),
                  "ATan2 Quadrant 3");
  DAX_TEST_ASSERT(test_equal(dax::exec::math::ATan2(-1.0, 1.0),
                             dax::Scalar(-0.25*dax::exec::math::Pi())),
                  "ATan2 Quadrant 4");
}

template<typename VectorType>
void TestTriangle(VectorType angle,
                  VectorType opposite,
                  VectorType adjacent,
                  VectorType hypotenuse)
{
  std::cout << "Testing triangle "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;

  DAX_TEST_ASSERT(test_equal(dax::exec::math::Sin(angle), opposite/hypotenuse),
                  "Sin failed test.");
  DAX_TEST_ASSERT(test_equal(dax::exec::math::Cos(angle), adjacent/hypotenuse),
                  "Cos failed test.");
  DAX_TEST_ASSERT(test_equal(dax::exec::math::Tan(angle), opposite/adjacent),
                  "Tan failed test.");

  DAX_TEST_ASSERT(test_equal(dax::exec::math::ASin(opposite/hypotenuse), angle),
                  "Arc Sin failed test.");
  DAX_TEST_ASSERT(test_equal(dax::exec::math::ACos(adjacent/hypotenuse), angle),
                  "Arc Cos failed test.");
  DAX_TEST_ASSERT(test_equal(dax::exec::math::ATan(opposite/adjacent), angle),
                  "Arc Tan failed test.");
}

template<typename VectorType>
void TestHyperbolic(VectorType x)
{
  std::cout << "Testing hyperbolic "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;

  VectorType zero;
  dax::exec::VectorFill(zero, dax::Scalar(0.0));
  VectorType minusX = zero - x;

  using namespace dax::exec::math;
  DAX_TEST_ASSERT(test_equal(SinH(x), 0.5f*(Exp(x) - Exp(minusX))),
                  "SinH does not match definition.");
  DAX_TEST_ASSERT(test_equal(CosH(x), 0.5f*(Exp(x) + Exp(minusX))),
                  "SinH does not match definition.");
  DAX_TEST_ASSERT(test_equal(TanH(x), SinH(x)/CosH(x)),
                  "TanH does not match definition");

  DAX_TEST_ASSERT(test_equal(ASinH(SinH(x)), x),
                  "SinH not inverting.");
  DAX_TEST_ASSERT(test_equal(ACosH(CosH(x)), x),
                  "CosH not inverting.");
  DAX_TEST_ASSERT(test_equal(ATanH(TanH(x)), x),
                  "TanH not inverting.");
}

void TestTrig()
{
  TestPi();
  TestArcTan2();
  TestTriangle(dax::Scalar(0.643501108793284),
               dax::Scalar(3.0),
               dax::Scalar(4.0),
               dax::Scalar(5.0));
  const dax::Scalar pi = dax::exec::math::Pi();
  TestTriangle(dax::make_Vector3(0.25*pi, 0.16666666667*pi, 0.33333333333*pi),
               dax::make_Vector3(1.0, 1.0, dax::exec::math::Sqrt(3.0)),
               dax::make_Vector3(1.0, dax::exec::math::Sqrt(3.0), 1.0),
               dax::make_Vector3(dax::exec::math::Sqrt(2.0), 2.0, 2.0));
  TestHyperbolic(dax::Scalar(0.5));
  TestHyperbolic(dax::make_Vector4(0.0, 0.25, 1.0, 2.0));
}

} // anonymous namespace

int UnitTestMathTrig(int, char *[])
{
  return dax::internal::Testing::Run(TestTrig);
}
