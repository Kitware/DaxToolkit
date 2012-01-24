/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/ArrayHandle.h>

#include <dax/cont/internal/Testing.h>

namespace
{
const dax::Id ARRAY_SIZE = 10;

void TestArrayHandle()
{
  dax::Scalar array[ARRAY_SIZE];

  // Create an array handle.
  dax::cont::ArrayHandle<dax::Scalar>
      arrayHandle(&array[0], &array[ARRAY_SIZE]);

  DAX_TEST_ASSERT(arrayHandle.GetNumberOfEntries() == ARRAY_SIZE,
                  "ArrayHandle has wrong number of entries.");

  DAX_TEST_ASSERT(arrayHandle.IsControlArrayValid(),
                  "Control data not valid.");

  // Make sure that invalidating any copy will invalidate all copies.
  dax::cont::ArrayHandle<dax::Scalar> arrayHandleCopy;
  arrayHandleCopy = arrayHandle;
  arrayHandleCopy.InvalidateControlArray();
  DAX_TEST_ASSERT(!arrayHandle.IsControlArrayValid(),
                  "Invalidate did not propagate to copies.");
}

}


int UnitTestArrayHandle(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestArrayHandle);
}
