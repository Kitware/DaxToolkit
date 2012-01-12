/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <dax/cont/UniformGrid.h>

namespace
{
const dax::Id ARRAY_SIZE = 10;

#define test_assert(condition, message) \
  test_assert_impl(condition, message, __FILE__, __LINE__);

static inline void test_assert_impl(bool condition,
                                    const std::string& message,
                                    const char *file,
                                    int line)
{
  if(!condition)
    {
    std::stringstream error;
    error << file << ":" << line << std::endl;
    error << message << std::endl;
    throw error.str();
    }
}

//TODO: Make tests for ExecutionPackageField* classes.

void TestExecutionPackageGrid()
{
  std::cout << "Test package UniformGrid." << std::endl;
  dax::cont::UniformGrid uniformGrid;
  dax::Id3 minExtent = dax::make_Id3(-ARRAY_SIZE, -ARRAY_SIZE, -ARRAY_SIZE);
  dax::Id3 maxExtent = dax::make_Id3(ARRAY_SIZE, ARRAY_SIZE, ARRAY_SIZE);
  uniformGrid.SetExtent(minExtent, maxExtent);
  dax::cont::internal::ExecutionPackageGrid<dax::cont::UniformGrid>
      uniformGridPackage(uniformGrid);

  dax::internal::StructureUniformGrid uniformStructure
      = uniformGridPackage.GetExecutionObject();
  test_assert(uniformStructure.Extent.Min == minExtent,
              "Bad uniform grid structure");
  test_assert(uniformStructure.Extent.Max == maxExtent,
              "Bad uniform grid structure");
}

} // Anonymous namespace

int UnitTestExecutionPackage(int, char *[])
{
  try
    {
    TestExecutionPackageGrid();
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
