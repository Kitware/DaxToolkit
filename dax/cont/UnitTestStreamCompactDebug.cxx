/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/ScheduleDebug.h>
#include <dax/cont/StreamCompactDebug.h>

#include <iostream>
#include <vector>

namespace {

const dax::Id ARRAY_SIZE = 500;

const dax::Id OFFSET = 1000;

}

namespace {

struct InitArray
{
  DAX_EXEC_EXPORT void operator()(std::vector<dax::Id> &array, dax::Id index)
  {
    array[index] = OFFSET + index;
  }
};

struct MakeMask
{
  DAX_EXEC_EXPORT void operator()(std::vector<dax::Id> &array, dax::Id index)
  {
    array[index] = index%2;
  }
};

}

int TestCompact()
  {
  //test the version of compact that takes in input and uses it as a stencil
  std::vector<dax::Id> array(ARRAY_SIZE);
  std::vector<dax::Id> result;

  //construct the index array
  dax::cont::scheduleDebug(MakeMask(), array, ARRAY_SIZE);
  dax::cont::streamCompactDebug(array,result);

  if(result.size() != array.size()/2)
    {
    std::cout << "Compacted array was of invalid length" << std::endl;
    return 1;
    }

  std::cout << "Checking results." << std::endl;
  for (dax::Id index = 0; index < result.size(); index++)
    {
    dax::Id value = result[index];
    if (value != index + OFFSET)
      {
      std::cout << "Got bad value." << std::endl;
      return 1;
      }
    }

  }

bool TestCompactWithStencil()
  {

  }

int UnitTestStreamCompactDebug(int, char *[])
{

  if(!TestCompact())
    {
    std::cout << "Stream Compact without a stencil array failed" << std::endl;
    return 1;
    }
  if(!TestCompactWithStencil())
    {
    std::cout << "Stream Compact with a stencil array failed" << std::endl;
    return 1;
    }
  return 0;
}
