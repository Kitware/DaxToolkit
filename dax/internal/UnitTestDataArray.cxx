/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/internal/DataArray.h>
#include <dax/Types.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

const dax::Id NUM_ENTRIES = 10;

#define TEST_FAIL(msg)                                  \
  {                                                     \
    std::stringstream error;                            \
    error << __FILE__ << ":" << __LINE__ << std::endl;  \
    error msg;                                          \
    throw error.str();                                  \
  }

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
    if (dataArray.GetValue(i) != i)
      {
      TEST_FAIL(<< "Bad data array.");
      }
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
    if (   (value.x != 3*i+0)
        || (value.y != 3*i+1)
        || (value.z != 3*i+2) )
      {
      TEST_FAIL(<< "Bad data array.");
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
    if (dataArray.GetValue(i) != i)
      {
      TEST_FAIL(<< "Bad data array.");
      }
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
    if (   (value.x != 3*i+0)
        || (value.y != 3*i+1)
        || (value.z != 3*i+2) )
      {
      TEST_FAIL(<< "Bad data array.");
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
    if (   (value.x != 4*i+0)
        || (value.y != 4*i+1)
        || (value.z != 4*i+2)
        || (value.w != 4*i+3) )
      {
      TEST_FAIL(<< "Bad data array.");
      }
    }
}

int UnitTestDataArray(int, char *[])
{
  try
    {
    TestIdArray();
    TestId3Array();
    TestScalarArray();
    TestVector3Array();
    TestVector4Array();
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
