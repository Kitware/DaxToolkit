/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cuda_exec_internal_CudaThreadIterator_h
#define __dax_cuda_exec_internal_CudaThreadIterator_h

#include <dax/Types.h>

namespace dax {
namespace cuda {
namespace exec {
namespace internal {

/// A simple iterator class that will iterate a thread over all of the indices
/// it is supposed to visit given the allocated thread blocks.
///
class CudaThreadIterator
{
public:
  DAX_EXEC_EXPORT CudaThreadIterator(dax::Id endIndex)
    : Index((blockIdx.x * blockDim.x) + threadIdx.x),
      Increment(gridDim.x * blockDim.x),
      EndIndex(endIndex) { }

  DAX_EXEC_EXPORT dax::Id GetIndex() const { return this->Index; }

  DAX_EXEC_EXPORT bool IsDone() const { return (this->Index >= this->EndIndex); }

  DAX_EXEC_EXPORT void Next() { this->Index += this->Increment; }

private:
  dax::Id Index;
  dax::Id Increment;
  dax::Id EndIndex;
};

}
}
}
}

#endif //__dax_cuda_exec_internal_CudaThreadIterator_h
