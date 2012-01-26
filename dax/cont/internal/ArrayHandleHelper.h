#ifndef __dax_cont_internal_ArrayHandleHelper_h
#define __dax_cont_internal_ArrayHandleHelper_h

#include <dax/cont/ArrayHandle.h>

namespace dax {
namespace cont {
namespace internal {

class ArrayHandleHelper
  {
public:
  template<typename T, class DeviceAdapter>
  inline static const typename DeviceAdapter::template ArrayContainerExecution<T>& ExecutionArray(
      const dax::cont::ArrayHandle<T,DeviceAdapter>& handle)
    {
    return handle.GetExecutionArray();
    }

  template<typename T, class DeviceAdapter>
  inline static typename DeviceAdapter::template ArrayContainerExecution<T>& ExecutionArray(
      dax::cont::ArrayHandle<T,DeviceAdapter>& handle)
    {
    return handle.GetExecutionArray();
    }

  template<typename T, class DeviceAdapter>
  inline static void UpdateArraySize(dax::cont::ArrayHandle<T,DeviceAdapter>& handle)
    {
    handle.UpdateArraySize();
    }

  };



} //internal
} //cont
} //dax




#endif // __dax_cont_internal_ArrayHandleHelper_h
