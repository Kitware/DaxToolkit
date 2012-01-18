/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_internal_DeviceAdapterError_h
#define __dax_cont_internal_DeviceAdapterError_h

#ifdef DAX_DEFAULT_DEVICE_ADAPTER
#undef DAX_DEFAULT_DEVICE_ADAPTER
#endif

#define DAX_DEFAULT_DEVICE_ADAPTER ::dax::cont::internal::DeviceAdapterError

namespace dax {
namespace cont {
namespace internal {

/// This is an invalid DeviceAdapter. The point of this class is to include the
/// header file to make this invalid class the default DeviceAdapter. From that
/// point, you have to specify an appropriate DeviceAdapter or else get a
/// compile error.
///
template<typename T = void>
struct DeviceAdapterError
{
  // Not implemented.
};

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_DeviceAdapterError_h
