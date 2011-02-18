/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxDataObject_h
#define __daxDataObject_h

#include "daxObject.h"

/// daxDataObject is the base-class for all data in the framework. It's an
/// abstract base-class defining the API.
class daxDataObject : public daxObject
{
public:
  daxDataObject();
  virtual ~daxDataObject();
  daxTypeMacro(daxDataObject, daxObject);

  /// API to get the raw data pointer and its size for reading or writing.
  virtual const void* GetDataPointer(const char* data_array_name) const = 0;

  /// API to get the raw data pointer and its size for reading or writing.
  virtual size_t GetDataSize(const char* data_array_name) const = 0;

  /// API to get the raw data pointer and its size for reading or writing.
  virtual void* GetWriteDataPointer(const char* data_array_name) = 0;

  /// OpenCL specific API to get the data-implementation code for the device.
  virtual const char* GetCode() const =0;

  /// These are used to obtain the host data-structure for \c opaque_data_type
  /// instance that's passed to the OpenCL kernel.
  virtual const void* GetOpaqueDataPointer() const =0;

  /// These are used to obtain the host data-structure for \c opaque_data_type
  /// instance that's passed to the OpenCL kernel.
  virtual size_t GetOpaqueDataSize() const=0;
};

daxDefinePtrMacro(daxDataObject);

#endif
