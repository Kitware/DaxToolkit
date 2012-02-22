/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_ReIndexDebug_h
#define __dax_cont_ReIndexDebug_h

#include <dax/Types.h>
#include <dax/internal/DataArray.h>
#include <dax/cont/ErrorExecution.h>
#include <dax/exec/internal/ErrorHandler.h>

#include <dax/cont/internal/ArrayContainerExecutionCPU.h>

#include <map>
#include <algorithm>
#include <vector>

namespace dax {
namespace cont {

DAX_CONT_EXPORT void ReIndexDebug(dax::internal::DataArray<dax::Id> ids)
{
  dax::Id size = ids.GetNumberOfEntries();
  dax::Id max = 1 + *(std::max_element(ids.GetPointer(),ids.GetPointer()+size));

  //create a bit vector of point usage
  std::vector<bool> pointUsage(max);
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
