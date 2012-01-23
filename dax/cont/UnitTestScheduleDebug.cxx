/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/ScheduleDebug.h>

#include <dax/exec/internal/ErrorHandler.h>

#include <iostream>

namespace {

const dax::Id ARRAY_SIZE = 500;

const dax::Id OFFSET = 1000;

}

namespace {

struct ClearArray
{
  DAX_EXEC_EXPORT void operator()(dax::Id *array,
                                  dax::Id index,
                                  dax::exec::internal::ErrorHandler &)
  {
    array[index] = OFFSET;
  }
};

struct AddArray
{
  DAX_EXEC_EXPORT void operator()(dax::Id *array,
                                  dax::Id index,
                                  dax::exec::internal::ErrorHandler &)
  {
    array[index] += index;
  }
};

const char ERROR_MESSAGE[] = "Got an error.";

struct OneError
{
  DAX_EXEC_EXPORT void operator()(
      dax::Id *, dax::Id index, dax::exec::internal::ErrorHandler &errorHandler)
  {
    if (index == ARRAY_SIZE/2)
      {
      errorHandler.RaiseError(ERROR_MESSAGE);
      }
  }
};

struct AllError
{
  DAX_EXEC_EXPORT void operator()(
      dax::Id *, dax::Id, dax::exec::internal::ErrorHandler &errorHandler)
  {
    errorHandler.RaiseError(ERROR_MESSAGE);
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

  std::cout << "Generating one error." << std::endl;
  char *message;
  message = dax::cont::scheduleDebug(OneError(), array, ARRAY_SIZE);
  if (strcmp(message, ERROR_MESSAGE) != 0)
    {
    std::cout << "Did not get expected error message." << std::endl;
    }

  std::cout << "Generating lots of errors." << std::endl;
  message = dax::cont::scheduleDebug(AllError(), array, ARRAY_SIZE);
  if (strcmp(message, ERROR_MESSAGE) != 0)
    {
    std::cout << "Did not get expected error message." << std::endl;
    }

  return 0;
}
