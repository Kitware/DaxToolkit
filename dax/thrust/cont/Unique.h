/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_thrust_cont_Unique_h
#define __dax_thrust_cont_Unique_h

#include <dax/Types.h>
#include <dax/thrust/cont/internal/ArrayContainerExecutionThrust.h>

#include <thrust/unique.h>

namespace dax {
namespace thrust {
namespace cont {


template<typename T>
DAX_CONT_EXPORT void unique(
    dax::thrust::cont::internal::ArrayContainerExecutionThrust<T> &values)
{
  typedef typename dax::thrust::cont::internal::ArrayContainerExecutionThrust<T>::iterator resultType;

  resultType newEnd = ::thrust::unique(values.GetBeginThrustIterator(),
                          values.GetEndThrustIterator());
  values.Allocate(::thrust::distance(values.GetBeginThrustIterator(),newEnd) );
}

}
}
} // namespace dax::thrust::cont

#endif //__dax_thrust_cont_Unique_h
