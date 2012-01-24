/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// This meta-test makes sure that the testing environment is properly reporting
// errors.

#include <dax/internal/Testing.h>

namespace {

void Fail()
{
  DAX_TEST_FAIL("I expect this error.");
}

void BadAssert()
{
  DAX_TEST_ASSERT(0 == 1, "I expect this error.");
}

void GoodAssert()
{
  DAX_TEST_ASSERT(1 == 1, "Always true.");
}

} // anonymous namespace

int UnitTestTesting(int, char *[])
{
  std::cout << "This call should fail." << std::endl;
  if (dax::internal::Testing::Run(Fail) == 0)
    {
    std::cout << "Did not get expected fail!" << std::endl;
    return 1;
    }
  std::cout << "This call should fail." << std::endl;
  if (dax::internal::Testing::Run(BadAssert) == 0)
    {
    std::cout << "Did not get expected fail!" << std::endl;
    return 1;
    }

  std::cout << "This call should pass." << std::endl;
  // This is what your main function typically looks like.
  return dax::internal::Testing::Run(GoodAssert);
}
