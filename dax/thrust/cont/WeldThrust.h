/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_thrust_cont_WeldThrust_h
#define __dax_thrust_cont_WeldThrust_h

#include <dax/Types.h>
#include <dax/thrust/cont/internal/ArrayContainerExecutionThrust.h>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace dax {
namespace thrust {
namespace cont {

template<typename T>
DAX_CONT_EXPORT void WeldThrust(dax::internal::DataArray<T> values)
{
  typedef typename ::thrust::device_vector<T>::iterator uniqueResultType;

  dax::Id size = values.GetNumberOfEntries();
  //create a device vector that is a copy of the ids array
  ::thrust::device_vector<T> uniqueValues(values.GetNumberOfEntries());
  ::thrust::device_ptr<T> raw_values =
      ::thrust::device_pointer_cast(values.GetPointer());
  ::thrust::copy(raw_values,raw_values+size,uniqueValues.begin());

  //sort the ids
  ::thrust::sort(uniqueValues.begin(),uniqueValues.end());

  // find unique items and erase redundancies
  uniqueResultType newEnd = ::thrust::unique(uniqueValues.begin(),
                                             uniqueValues.end());

  // find index of each input vertex in the list of unique vertices
  ::thrust::lower_bound(uniqueValues.begin(), newEnd,
                        raw_values,raw_values+size,
                        raw_values);
}

}
}
} // namespace dax::thrust::cont

#endif //__dax_thrust_cont_WeldThrust_h
