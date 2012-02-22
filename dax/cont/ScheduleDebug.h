/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_Schedule_h
#define __dax_cont_Schedule_h

#include <dax/Types.h>
#include <dax/internal/DataArray.h>
#include <dax/cont/ErrorExecution.h>
#include <dax/exec/internal/ErrorHandler.h>

#include <map>
#include <algorithm>
#include <vector>

namespace dax {
namespace cont {

namespace internal {

DAX_CONT_EXPORT dax::internal::DataArray<char> getScheduleDebugErrorArray()
{
  const dax::Id ERROR_ARRAY_SIZE = 1024;
  static char ErrorArrayBuffer[ERROR_ARRAY_SIZE];
  dax::internal::DataArray<char> ErrorArray(ErrorArrayBuffer, ERROR_ARRAY_SIZE);
  return ErrorArray;
}

}

template<class Functor, class Parameters>
DAX_CONT_EXPORT void scheduleDebug(Functor functor,
                                   Parameters parameters,
                                   dax::Id numInstances)
{
  dax::internal::DataArray<char> errorArray
      = internal::getScheduleDebugErrorArray();

  // Clear error value.
  errorArray.SetValue(0, '\0');

  dax::exec::internal::ErrorHandler errorHandler(errorArray);

  for (dax::Id index = 0; index < numInstances; index++)
    {
    functor(parameters, index, errorHandler);
    if (errorHandler.IsErrorRaised())
      {
      throw dax::cont::ErrorExecution(errorArray.GetPointer());
      }
    }
}


template<class Functor, class Parameters>
DAX_CONT_EXPORT void scheduleDebug(Functor functor,
                                   Parameters parameters,
                                   dax::internal::DataArray<dax::Id> ids)
{
  dax::internal::DataArray<char> errorArray
      = internal::getScheduleDebugErrorArray();

  // Clear error value.
  errorArray.SetValue(0, '\0');

  dax::exec::internal::ErrorHandler errorHandler(errorArray);
  dax::Id size = ids.GetNumberOfEntries();
  for (dax::Id index = 0; index < size; index++)
    {
    functor(parameters, ids.GetValue(index), errorHandler);
    if (errorHandler.IsErrorRaised())
      {
      throw dax::cont::ErrorExecution(errorArray.GetPointer());
      }
    }
}

template<class Functor, class Parameters>
DAX_CONT_EXPORT void reindexDebug(dax::internal::DataArray<dax::Id> ids)
{
  dax::Id size = ids.GetNumberOfEntries();
  dax::Id max = std::max_element(ids.GetPointer(),ids.GetPointer()+size);

  //create a bit vector of point usage
  std::vector<bool> pointUsage(max+1);
  for(dax::Id i=0; i < size; ++i)
    {
    //flag each point id has being used
    pointUsage[ids.GetValue(i)]=true;
    }

  std::map<dax::Id,dax::Id> uniquePointMap;
  dax::Id newId = 0;
  for(dax::Id i=0; i < max; ++i)
    {
    if(pointUsage[i])
      {
      uniquePointMap.insert(std::pair<dax::Id,dax::Id>(i,newId++));
      }
    }

  //reindex the passed in array
  for(dax::Id i=0; i < size; ++i)
    {
    ids.SetValue(i, uniquePointMap.find(ids.GetValue(i))->second);
    }
}

}
} // namespace dax::cont

#endif //__dax_cont_Schedule_h
