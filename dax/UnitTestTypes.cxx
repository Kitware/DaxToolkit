/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/Types.h>

#include <dax/internal/Testing.h>

namespace {

static void TestVector3()
{
  dax::Vector3 a = dax::make_Vector3(2, 4, 6);
  dax::Vector3 b = dax::make_Vector3(1, 2, 3);
  dax::Scalar s = 5;

  dax::Vector3 plus = a + b;
  if ((plus[0] != 3) || (plus[1] != 6) || (plus[2] != 9))
    {
    DAX_TEST_FAIL("Vectors do not add correctly.");
    }

  dax::Vector3 minus = a - b;
  if ((minus[0] != 1) || (minus[1] != 2) || (minus[2] != 3))
    {
    DAX_TEST_FAIL("Vectors to not subtract correctly.");
    }

  dax::Vector3 mult = a * b;
  if ((mult[0] != 2) || (mult[1] != 8) || (mult[2] != 18))
    {
    DAX_TEST_FAIL("Vectors to not multiply correctly.");
    }

  dax::Vector3 div = a / b;
  if ((div[0] != 2) || (div[1] != 2) || (div[2] != 2))
    {
    DAX_TEST_FAIL("Vectors to not divide correctly.");
    }

  mult = s * a;
  if ((mult[0] != 10) || (mult[1] != 20) || (mult[2] != 30))
    {
    DAX_TEST_FAIL("Vector and scalar to not multiply correctly.");
    }

  mult = a * s;
  if ((mult[0] != 10) || (mult[1] != 20) || (mult[2] != 30))
    {
    DAX_TEST_FAIL("Vector and scalar to not multiply correctly.");
    }

  if (a == b)
    {
    DAX_TEST_FAIL("operator== wrong");
    }
  if (!(a != b))
    {
    DAX_TEST_FAIL("operator!= wrong");
    }

  if (dax::dot(a, b) != 28)
    {
    DAX_TEST_FAIL("dot(Vector3) wrong");
    }
}

static void TestVector4()
{
  dax::Vector4 a = dax::make_Vector4(2, 4, 6, 8);
  dax::Vector4 b = dax::make_Vector4(1, 2, 3, 4);
  dax::Scalar s = 5;

  dax::Vector4 plus = a + b;
  if ((plus[0] != 3) || (plus[1] != 6) || (plus[2] != 9) || (plus[3] != 12))
    {
    DAX_TEST_FAIL("Vectors do not add correctly.");
    }

  dax::Vector4 minus = a - b;
  if ((minus[0] != 1) || (minus[1] != 2) || (minus[2] != 3) || (minus[3] != 4))
    {
    DAX_TEST_FAIL("Vectors to not subtract correctly.");
    }

  dax::Vector4 mult = a * b;
  if ((mult[0] != 2) || (mult[1] != 8) || (mult[2] != 18) || (mult[3] != 32))
    {
    DAX_TEST_FAIL("Vectors to not multiply correctly.");
    }

  dax::Vector4 div = a / b;
  if ((div[0] != 2) || (div[1] != 2) || (div[2] != 2) || (div[3] != 2))
    {
    DAX_TEST_FAIL("Vectors to not divide correctly.");
    }

  mult = s * a;
  if ((mult[0] != 10) || (mult[1] != 20) || (mult[2] != 30) || (mult[3] != 40))
    {
    DAX_TEST_FAIL("Vector and scalar to not multiply correctly.");
    }

  mult = a * s;
  if ((mult[0] != 10) || (mult[1] != 20) || (mult[2] != 30) || (mult[3] != 40))
    {
    DAX_TEST_FAIL("Vector and scalar to not multiply correctly.");
    }

  if (a == b)
    {
    DAX_TEST_FAIL("operator== wrong");
    }
  if (!(a != b))
    {
    DAX_TEST_FAIL("operator!= wrong");
    }

  if (dax::dot(a, b) != 60)
    {
    DAX_TEST_FAIL("dot(Vector4) wrong");
    }
}

static void TestId3()
{
  dax::Id3 a = dax::make_Id3(2, 4, 6);
  dax::Id3 b = dax::make_Id3(1, 2, 3);
  dax::Id s = 5;

  dax::Id3 plus = a + b;
  if ((plus[0] != 3) || (plus[1] != 6) || (plus[2] != 9))
    {
    DAX_TEST_FAIL("Vectors do not add correctly.");
    }

  dax::Id3 minus = a - b;
  if ((minus[0] != 1) || (minus[1] != 2) || (minus[2] != 3))
    {
    DAX_TEST_FAIL("Vectors to not subtract correctly.");
    }

  dax::Id3 mult = a * b;
  if ((mult[0] != 2) || (mult[1] != 8) || (mult[2] != 18))
    {
    DAX_TEST_FAIL("Vectors to not multiply correctly.");
    }

  dax::Id3 div = a / b;
  if ((div[0] != 2) || (div[1] != 2) || (div[2] != 2))
    {
    DAX_TEST_FAIL("Vectors to not divide correctly.");
    }

  mult = s * a;
  if ((mult[0] != 10) || (mult[1] != 20) || (mult[2] != 30))
    {
    DAX_TEST_FAIL("Vector and scalar to not multiply correctly.");
    }

  mult = a * s;
  if ((mult[0] != 10) || (mult[1] != 20) || (mult[2] != 30))
    {
    DAX_TEST_FAIL("Vector and scalar to not multiply correctly.");
    }

  if (a == b)
    {
    DAX_TEST_FAIL("operator== wrong");
    }
  if (!(a != b))
    {
    DAX_TEST_FAIL("operator!= wrong");
    }

  if (dax::dot(a, b) != 28)
    {
    DAX_TEST_FAIL("dot(Id3) wrong");
    }
}

void TestTypes()
{
  TestVector3();
  TestVector4();
  TestId3();
}

} // anonymous namespace

int UnitTestTypes(int, char *[])
{
  return dax::internal::Testing::Run(TestTypes);
}
