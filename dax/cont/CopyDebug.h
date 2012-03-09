/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_CopyDebug_h
#define __dax_cont_CopyDebug_h

#include <dax/cont/internal/ArrayContainerExecutionCPU.h>
#include <algorithm>

namespace dax {
namespace cont {

template<typename T>
DAX_CONT_EXPORT void copyDebug(
    const dax::cont::internal::ArrayContainerExecutionCPU<T> &from,
    dax::cont::internal::ArrayContainerExecutionCPU<T> &to)
{
  std::copy(from.begin(),from.end(),to.begin());
}

}
} // namespace dax::cont

#endif //__dax_cont_CopyDebug_h
