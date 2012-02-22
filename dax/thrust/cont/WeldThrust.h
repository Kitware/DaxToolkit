/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_thrust_cont_WeldThrust_h
#define __dax_thrust_cont_WeldThrust_h

#include <dax/Types.h>
#include <dax/thrust/cont/internal/ArrayContainerExecutionThrust.h>

#include <dax/thrust/cont/ScheduleThrust.h>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace dax {
namespace thrust {
namespace cont {
DAX_CONT_EXPORT void WeldThrust(dax::internal::DataArray<dax::Id> ids)
{
  typedef ::thrust::device_vector<dax::Id>::iterator uniqueResultType;

  dax::Id size = ids.GetNumberOfEntries();
  //create a device vector that is a copy of the ids array
  ::thrust::device_vector<dax::Id> lookup(ids.GetNumberOfEntries());
  ::thrust::device_ptr<dax::Id> raw_ids =
      ::thrust::device_pointer_cast(ids.GetPointer());
  ::thrust::copy(raw_ids,raw_ids+size,lookup.begin());

  //sort the ids
  ::thrust::sort(lookup.begin(),lookup.end());

  // find unique items and erase redundancies
  uniqueResultType newEnd = ::thrust::unique(lookup.begin(), lookup.end());

  // find index of each input vertex in the list of unique vertices
  ::thrust::lower_bound(lookup.begin(), newEnd,
                        raw_ids,raw_ids+size,
                        raw_ids);
}

}
}
} // namespace dax::thrust::cont

#endif //__dax_thrust_cont_WeldThrust_h
