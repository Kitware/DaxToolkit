/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/internal/DataArray.h>
#include <dax/Types.h>
#include <dax/exec/VectorOperations.h>

#include <dax/internal/Testing.h>

namespace {

struct TestArray
{
  static const dax::Id NUM_ENTRIES = 10;

  template<typename T> void FillArray(T *data, dax::Id num) const
  {
    for (dax::Id index = 0; index < num; index++)
      {
      dax::exec::VectorFill(data[index], index);
      }
  }

  template<typename T> void operator()(const T&) const
  {
    T data[NUM_ENTRIES];
    this->FillArray(data, NUM_ENTRIES);

    dax::internal::DataArray<T> array(data, NUM_ENTRIES);

    std::cout << "  checking" << std::endl;
    for (dax::Id index = 0; index < NUM_ENTRIES; index++)
      {
      DAX_TEST_ASSERT(test_equal(array.GetValue(index),
                                 dax::exec::VectorFill<T>(index)),
                      "Bad data array");
      }
  }
};

void DataArrayTests()
{
  dax::internal::Testing::TryAllTypes(TestArray());
}

} // anonymous namespace

int UnitTestDataArray(int, char *[])
{
  return dax::internal::Testing::Run(DataArrayTests);
}
