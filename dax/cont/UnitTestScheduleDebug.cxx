/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/ScheduleDebug.h>

#include <iostream>

namespace {

const dax::Id ARRAY_SIZE = 500;

const dax::Id OFFSET = 1000;

}

namespace {

struct ClearArray
{
  DAX_EXEC_EXPORT void operator()(dax::Id *array, dax::Id index)
  {
    array[index] = OFFSET;
  }
};

struct AddArray
{
  DAX_EXEC_EXPORT void operator()(dax::Id *array, dax::Id index)
  {
    array[index] += index;
  }
};

}

int UnitTestScheduleDebug(int, char *[])
{
  dax::Id array[ARRAY_SIZE];

  std::cout << "Running clear." << std::endl;
  dax::cont::scheduleDebug(ClearArray(), array, ARRAY_SIZE);

  std::cout << "Running add." << std::endl;
  dax::cont::scheduleDebug(AddArray(), array, ARRAY_SIZE);

  std::cout << "Checking results." << std::endl;
  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    dax::Id value = array[index];
    if (value != index + OFFSET)
      {
      std::cout << "Got bad value." << std::endl;
      return 1;
      }
    }

  return 0;
}
