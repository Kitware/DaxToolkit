/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_LowerBoundsDebug_h
#define __dax_cont_LowerBoundsDebug_h

#include <dax/Types.h>
#include <dax/cont/internal/ArrayContainerExecutionCPU.h>

#include <algorithm>

namespace dax {
namespace cont {
template<typename T>
DAX_CONT_EXPORT void lowerBoundsDebug(
    const dax::cont::internal::ArrayContainerExecutionCPU<T>& input,
    const dax::cont::internal::ArrayContainerExecutionCPU<T>& values,
    const dax::cont::internal::ArrayContainerExecutionCPU<dax::Id>& output)
{
  typedef typename dax::cont::internal::ArrayContainerExecutionCPU<T>::const_iterator CIter;
  typedef typename dax::cont::internal::ArrayContainerExecutionCPU<dax::Id>::iterator OIter;

  //stl lower_bound only supports a single value to search for.
  //So we iterate over all the values and search for each one
  OIter out=output.begin();
  for(CIter i=values.begin(); i!=values.end(); ++i,++out)
    {
    *out = std::lower_bound(input.begin(),input.end(),*i);
    }
}


}
} // namespace dax::cont

#endif //__dax_cont_LowerBoundsDebug_h
