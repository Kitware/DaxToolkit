/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/internal/DataArray.h>
#include <dax/Types.h>

#include <dax/internal/Testing.h>

namespace {

const dax::Id NUM_ENTRIES = 10;

template<typename T>
static void FillRawData(T *rawData, dax::Id num)
{
  for (int i = 0; i < num; i++)
    {
    rawData[i] = i;
    }
}

static void TestIdArray()
{
  dax::Id rawData[NUM_ENTRIES];
  FillRawData(rawData, NUM_ENTRIES);

  dax::internal::DataArray<dax::Id> dataArray
      = dax::internal::make_DataArrayId(rawData, NUM_ENTRIES);

  std::cout << "Checking DataArray<dax::Id>" << std::endl;
  for (int i = 0; i < NUM_ENTRIES; i++)
    {
    DAX_TEST_ASSERT(dataArray.GetValue(i) == i,
                    "Bad data array.");
    }
}

static void TestId3Array()
{
  dax::Id rawData[3*NUM_ENTRIES];
  FillRawData(rawData, 3*NUM_ENTRIES);

  dax::internal::DataArray<dax::Id3> dataArray
      = dax::internal::make_DataArrayId3(rawData, NUM_ENTRIES);

  std::cout << "Checking DataArray<dax::Id3>" << std::endl;
  for (int i = 0; i < NUM_ENTRIES; i++)
    {
    dax::Id3 value = dataArray.GetValue(i);
    if (   (value[0] != 3*i+0)
        || (value[1] != 3*i+1)
        || (value[2] != 3*i+2) )
      {
      DAX_TEST_FAIL("Bad data array.");
      }
    }
}

static void TestScalarArray()
{
  dax::Scalar rawData[NUM_ENTRIES];
  FillRawData(rawData, NUM_ENTRIES);

  dax::internal::DataArray<dax::Scalar> dataArray
      = dax::internal::make_DataArrayScalar(rawData, NUM_ENTRIES);

  std::cout << "Checking DataArray<dax::Scalar>" << std::endl;
  for (int i = 0; i < NUM_ENTRIES; i++)
    {
    DAX_TEST_ASSERT(dataArray.GetValue(i) == i,
                    "Bad data array.");
    }
}

static void TestVector3Array()
{
  dax::Scalar rawData[3*NUM_ENTRIES];
  FillRawData(rawData, 3*NUM_ENTRIES);

  dax::internal::DataArray<dax::Vector3> dataArray
      = dax::internal::make_DataArrayVector3(rawData, NUM_ENTRIES);

  std::cout << "Checking DataArray<dax::Vector3>" << std::endl;
  for (int i = 0; i < NUM_ENTRIES; i++)
    {
    dax::Vector3 value = dataArray.GetValue(i);
    if (   (value[0] != 3*i+0)
        || (value[1] != 3*i+1)
        || (value[2] != 3*i+2) )
      {
      DAX_TEST_FAIL("Bad data array.");
      }
    }
}

static void TestVector4Array()
{
  dax::Scalar rawData[4*NUM_ENTRIES];
  FillRawData(rawData, 4*NUM_ENTRIES);

  dax::internal::DataArray<dax::Vector4> dataArray
      = dax::internal::make_DataArrayVector4(rawData, NUM_ENTRIES);

  std::cout << "Checking DataArray<dax::Vector4>" << std::endl;
  for (int i = 0; i < NUM_ENTRIES; i++)
    {
    dax::Vector4 value = dataArray.GetValue(i);
    if (   (value[0] != 4*i+0)
        || (value[1] != 4*i+1)
        || (value[2] != 4*i+2)
        || (value[3] != 4*i+3) )
      {
      DAX_TEST_FAIL("Bad data array.");
      }
    }
}

void DataArrayTests()
{
  TestIdArray();
  TestId3Array();
  TestScalarArray();
  TestVector3Array();
  TestVector4Array();
}

} // anonymous namespace

int UnitTestDataArray(int, char *[])
{
  return dax::internal::Testing::Run(DataArrayTests);
}
