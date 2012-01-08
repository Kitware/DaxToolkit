/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_internal_ArrayContainerExecution_h
#define __dax_cont_internal_ArrayContainerExecution_h

#include <dax/Types.h>

// TODO: Come up with a better way to choose the appropriate implementation
// for ArrayContainerExecution.
#ifdef DAX_CUDA
#include <dax/cuda/cont/internal/ArrayContainerExecution.h>
namespace dax {
namespace cont {
namespace internal {
template<typename T>
class ArrayContainerExecution
    : public dax::cuda::cont::internal::ArrayContainerExecution<T>
{ };
}
}
}
#else

#include <dax/internal/DataArray.h>
#include <dax/cont/internal/IteratorContainer.h>

#include <assert.h>

namespace dax {
namespace cont {
namespace internal {

/// Manages an execution environment array, which may need to be allocated
/// on seperate device memory.
template<typename T>
class ArrayContainerExecution
{
public:
  typedef T ValueType;

  /// On inital creation, no memory is allocated on the device.
  ///
  ArrayContainerExecution() { }

  /// Allocates an array on the device large enough to hold the given number of
  /// entries.
  ///
  void Allocate(dax::Id) {
    assert("Unsupported execution array");
  }

  /// Copies the data pointed to by the passed in \c iterators (assumed to be
  /// in the control environment), into the array in the execution environment
  /// managed by this class.
  ///
  template<class IteratorType>
  void CopyFromControlToExecution(
      const dax::cont::internal::IteratorContainer<IteratorType> &) {
    assert("Unsupported execution array");
  }

  /// Copies the data from the array in the execution environment managed by
  /// this class into the memory passed in the \c iterators (assumed to be in
  /// the control environment).
  ///
  template<class IteratorType>
  void CopyFromExecutionToControl(
      const dax::cont::internal::IteratorContainer<IteratorType> &) {
    assert("Unsupported execution array");
  }

  /// Frees any resources (i.e. memory) on the device.
  ///
  void ReleaseResources() {
    assert("Unsupported execution array");
  }

  /// Gets a DataArray that is valid in the execution environment.
  dax::internal::DataArray<ValueType> GetExecutionArray() {
    assert("Unsupported execution array");
    return dax::internal::DataArray<ValueType>();
  }

private:
  ArrayContainerExecution(const ArrayContainerExecution &); // Not implemented
  void operator=(const ArrayContainerExecution &);          // Not implemented
};

}
}
}

#endif // DAX_CUDA

#endif //__dax_cont_internal_ArrayContainerExecution_h
