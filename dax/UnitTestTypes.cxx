/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/Types.h>

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

static void TestVector3()
{
  dax::Vector3 a = dax::make_Vector3(2, 4, 6);
  dax::Vector3 b = dax::make_Vector3(1, 2, 3);
  dax::Scalar s = 5;

  dax::Vector3 plus = a + b;
  if ((plus[0] != 3) || (plus[1] != 6) || (plus[2] != 9))
    {
    TEST_FAIL(<< "Vectors do not add correctly.");
    }

  dax::Vector3 minus = a - b;
  if ((minus[0] != 1) || (minus[1] != 2) || (minus[2] != 3))
    {
    TEST_FAIL(<< "Vectors to not subtract correctly.");
    }

  dax::Vector3 mult = a * b;
  if ((mult[0] != 2) || (mult[1] != 8) || (mult[2] != 18))
    {
    TEST_FAIL(<< "Vectors to not multiply correctly.");
    }

  dax::Vector3 div = a / b;
  if ((div[0] != 2) || (div[1] != 2) || (div[2] != 2))
    {
    TEST_FAIL(<< "Vectors to not divide correctly.");
    }

  mult = s * a;
  if ((mult[0] != 10) || (mult[1] != 20) || (mult[2] != 30))
    {
    TEST_FAIL(<< "Vector and scalar to not multiply correctly.");
    }

  mult = a * s;
  if ((mult[0] != 10) || (mult[1] != 20) || (mult[2] != 30))
    {
    TEST_FAIL(<< "Vector and scalar to not multiply correctly.");
    }

  if (a == b)
    {
    TEST_FAIL(<< "operator== wrong");
    }
  if (!(a != b))
    {
    TEST_FAIL(<< "operator!= wrong");
    }

  if (dax::dot(a, b) != 28)
    {
    TEST_FAIL(<< "dot(Vector3) wrong");
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
    TEST_FAIL(<< "Vectors do not add correctly.");
    }

  dax::Vector4 minus = a - b;
  if ((minus[0] != 1) || (minus[1] != 2) || (minus[2] != 3) || (minus[3] != 4))
    {
    TEST_FAIL(<< "Vectors to not subtract correctly.");
    }

  dax::Vector4 mult = a * b;
  if ((mult[0] != 2) || (mult[1] != 8) || (mult[2] != 18) || (mult[3] != 32))
    {
    TEST_FAIL(<< "Vectors to not multiply correctly.");
    }

  dax::Vector4 div = a / b;
  if ((div[0] != 2) || (div[1] != 2) || (div[2] != 2) || (div[3] != 2))
    {
    TEST_FAIL(<< "Vectors to not divide correctly.");
    }

  mult = s * a;
  if ((mult[0] != 10) || (mult[1] != 20) || (mult[2] != 30) || (mult[3] != 40))
    {
    TEST_FAIL(<< "Vector and scalar to not multiply correctly.");
    }

  mult = a * s;
  if ((mult[0] != 10) || (mult[1] != 20) || (mult[2] != 30) || (mult[3] != 40))
    {
    TEST_FAIL(<< "Vector and scalar to not multiply correctly.");
    }

  if (a == b)
    {
    TEST_FAIL(<< "operator== wrong");
    }
  if (!(a != b))
    {
    TEST_FAIL(<< "operator!= wrong");
    }

  if (dax::dot(a, b) != 60)
    {
    TEST_FAIL(<< "dot(Vector4) wrong");
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
    TEST_FAIL(<< "Vectors do not add correctly.");
    }

  dax::Id3 minus = a - b;
  if ((minus[0] != 1) || (minus[1] != 2) || (minus[2] != 3))
    {
    TEST_FAIL(<< "Vectors to not subtract correctly.");
    }

  dax::Id3 mult = a * b;
  if ((mult[0] != 2) || (mult[1] != 8) || (mult[2] != 18))
    {
    TEST_FAIL(<< "Vectors to not multiply correctly.");
    }

  dax::Id3 div = a / b;
  if ((div[0] != 2) || (div[1] != 2) || (div[2] != 2))
    {
    TEST_FAIL(<< "Vectors to not divide correctly.");
    }

  mult = s * a;
  if ((mult[0] != 10) || (mult[1] != 20) || (mult[2] != 30))
    {
    TEST_FAIL(<< "Vector and scalar to not multiply correctly.");
    }

  mult = a * s;
  if ((mult[0] != 10) || (mult[1] != 20) || (mult[2] != 30))
    {
    TEST_FAIL(<< "Vector and scalar to not multiply correctly.");
    }

  if (a == b)
    {
    TEST_FAIL(<< "operator== wrong");
    }
  if (!(a != b))
    {
    TEST_FAIL(<< "operator!= wrong");
    }

  if (dax::dot(a, b) != 28)
    {
    TEST_FAIL(<< "dot(Id3) wrong");
    }
}

int UnitTestTypes(int, char *[])
{
  try
    {
    TestVector3();
    TestVector4();
    TestId3();
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
