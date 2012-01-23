#ifndef __dax_cont_internal_ArrayHandleHelper_h
#define __dax_cont_internal_ArrayHandleHelper_h

#include <dax/cont/ArrayHandle.h>

namespace dax {
namespace cont {
namespace internal {

class Converter
  {
public:
  template<typename T, class DeviceAdapter>
  static const typename DeviceAdapter::template ArrayContainerExecution<T>& ExecutionArray(
      const dax::cont::ArrayHandle<T,DeviceAdapter>& handle)
    {
    return handle.GetExecutionArray();
    }

  };



} //internal
} //cont
} //dax




#endif // __dax_cont_internal_ArrayHandleHelper_h
